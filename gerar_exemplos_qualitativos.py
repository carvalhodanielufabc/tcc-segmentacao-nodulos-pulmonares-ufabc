# =============================================================================
# Título do Projeto: Comparação entre Método Clássico de Processamento Digital 
#                    de Imagens e Rede Neural para Segmentação de Nódulos 
#                    Pulmonares em Tomografias Computadorizadas
# Autores:           Daniel Augusto de Carvalho e Leonardo Reis dos Santos Melati
# Instituição:       UFABC (Universidade Federal do ABC)
# Ano:               2026
# Arquivo:           gerar_exemplos_qualitativos.py
# Descrição:         Script auxiliar para gerar painéis comparativos lado a lado 
#                    entre o Método Clássico, U-Net e o Ground Truth.
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader

# --- Configurações ---

CAMINHO_SAIDA = r"D:\TG\processadoV4"
CAMINHO_SLICES = r"D:\TG\slices_2dV5"
CAMINHO_SAIDA_MODELO = r"D:\TG\modelo_unetV5"
CAMINHO_SPLIT = os.path.join(CAMINHO_SAIDA, "split_pacientes_V5.csv")
CAMINHO_NODULOS = os.path.join(CAMINHO_SAIDA, "nodulos_validos_V5.csv")
CAMINHO_METRICAS_NODULO_CLASSICO = os.path.join(
    CAMINHO_SLICES, "metricas_por_nodulo_teste.csv")
CAMINHO_SAIDA_FIGURAS = os.path.join(CAMINHO_SAIDA_MODELO, "exemplos_qualitativos")

# Pacientes escolhidos, cobrindo diferentes cenários discutidos na 5.3.3:
#   - 0997, 1002: o clássico detectou 100% dos nódulos (bom exemplo "puro"
#     do método funcionando bem, sem depender do fallback)
#   - 0583, 0492: casos em que a U-Net superou claramente o clássico
#   - 0661, 0894: casos em que ambos os métodos tiveram mais dificuldade
PACIENTES_EXEMPLO = [
    "LIDC-IDRI-0997", "LIDC-IDRI-1002",
    "LIDC-IDRI-0583", "LIDC-IDRI-0492",
    "LIDC-IDRI-0661", "LIDC-IDRI-0894",
]

HU_MIN, HU_MAX = -1000, 400  # janela de exibição (mesma do script 2)

# --- Carrega os módulos dos scripts 5, 7 e 9, reaproveitando o mesmo
# código usado nas avaliações de teste, para garantir consistência ---

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_mod5 = SourceFileLoader(
    "metodo_classico", os.path.join(_SCRIPT_DIR, "5_metodo_classico_calibrar.py")
).load_module()
_mod7 = SourceFileLoader(
    "metodo_classico_por_nodulo",
    os.path.join(_SCRIPT_DIR, "7_metodo_classico_avaliar_por_nodulo.py")
).load_module()
_mod9 = SourceFileLoader(
    "unet_avaliar", os.path.join(_SCRIPT_DIR, "9_unet_avaliar_teste.py")
).load_module()


def janela_hu(slice_ct):
    clip = np.clip(slice_ct, HU_MIN, HU_MAX)
    return (clip - HU_MIN) / (HU_MAX - HU_MIN)


def normalizar_hu_classico(volume_hu):
    if hasattr(_mod5, "normalizar_hu"):
        return _mod5.normalizar_hu(volume_hu)
    clip = np.clip(volume_hu, _mod5.HU_MIN, _mod5.HU_MAX)
    return ((clip - _mod5.HU_MIN) / (_mod5.HU_MAX - _mod5.HU_MIN)).astype(np.float32)


def bbox_com_folga(mascara_2d, folga=25):
    coords = np.argwhere(mascara_2d)
    if len(coords) == 0:
        return slice(0, mascara_2d.shape[0]), slice(0, mascara_2d.shape[1])
    y0, x0 = coords[:, 0].min(), coords[:, 1].min()
    y1, x1 = coords[:, 0].max(), coords[:, 1].max()
    H, W = mascara_2d.shape
    y0 = max(0, y0 - folga); x0 = max(0, x0 - folga)
    y1 = min(H, y1 + folga); x1 = min(W, x1 + folga)
    return slice(y0, y1), slice(x0, x1)


def escolher_nodulo(paciente, cluster_ids, metricas_classico_df):
    """
    Escolhe um cluster_id (nódulo) para ilustrar o paciente:
      1) Prioriza um nódulo com 'encontrado'==True pelo método clássico,
         entre esses escolhe o de maior volume (mais fácil de visualizar).
      2) Se nenhum foi encontrado pelo clássico, cai para o de maior
         volume entre todos os válidos do paciente, e sinaliza isso.
    Retorna (cluster_id, foi_encontrado_pelo_classico: bool).
    """
    sub = metricas_classico_df[metricas_classico_df["paciente"] == paciente]
    encontrados = sub[sub["encontrado"] == True]  # noqa: E712

    if len(encontrados) > 0:
        linha = encontrados.sort_values("volume_gt_voxels", ascending=False).iloc[0]
        return int(linha["cluster_id"]), True

    if len(sub) > 0:
        linha = sub.sort_values("volume_gt_voxels", ascending=False).iloc[0]
        print("  [AVISO] %s: método clássico não encontrou NENHUM nódulo "
              "válido — usando o de maior volume (cluster %d) como exemplo "
              "de falha real, não um bug do script." % (
                  paciente, int(linha["cluster_id"])))
        return int(linha["cluster_id"]), False

    # Fallback final, caso o paciente nem apareça em metricas_por_nodulo:
    # usa o primeiro cluster_id válido do indice de nódulos.
    print("  [AVISO] %s: não encontrado em metricas_por_nodulo_teste.csv — "
          "usando o primeiro cluster_id válido disponível." % paciente)
    return int(cluster_ids[0]), False


def escolher_z_com_melhor_sobreposicao(mascara_gt, volume_norm_classico,
                                        volume_norm_unet, params_classico, modelo):
    """
    Roda o método clássico E a U-Net em TODOS os slices onde o nódulo
    (mascara_gt) está presente, e escolhe o slice que maximiza a
    sobreposição COMBINADA dos dois métodos com a referência. Isso evita
    escolher, por acaso, um slice que favoreça a visualização de um
    método e deixe o outro sem nenhuma região visível — já que a
    definição de "nódulo detectado" nos scripts 7 e 10 é 3D, não por
    slice, um único corte pode não coincidir com onde a sobreposição de
    cada método de fato acontece.

    Retorna (z_escolhido, dict_z_para_pred_classico, dict_z_para_pred_unet).
    """
    zs = np.where(mascara_gt.sum(axis=(1, 2)) > 0)[0]
    if len(zs) == 0:
        return None, {}, {}

    preds_classico_por_z = {}
    preds_unet_por_z = {}
    sobreposicao_combinada_por_z = {}

    for z in zs:
        pulmao = _mod5.segmentar_pulmao(volume_norm_classico[z])
        pred_c = _mod5.segmentar_nodulos_classico(
            volume_norm_classico[z], pulmao, **params_classico)
        pred_u = _mod9.prever_slice(modelo, volume_norm_unet[z]).astype(bool)

        preds_classico_por_z[z] = pred_c
        preds_unet_por_z[z] = pred_u

        overlap_c = np.logical_and(pred_c, mascara_gt[z]).sum()
        overlap_u = np.logical_and(pred_u, mascara_gt[z]).sum()
        sobreposicao_combinada_por_z[z] = overlap_c + overlap_u

    melhor_z = max(sobreposicao_combinada_por_z, key=sobreposicao_combinada_por_z.get)
    if sobreposicao_combinada_por_z[melhor_z] > 0:
        return int(melhor_z), preds_classico_por_z, preds_unet_por_z

    # Nenhum slice teve sobreposição em NENHUM dos dois métodos — cai
    # para o slice de maior área de referência (falha real de ambos).
    z_maior_area = int(zs[np.argmax(mascara_gt[zs].sum(axis=(1, 2)))])
    return z_maior_area, preds_classico_por_z, preds_unet_por_z


def salvar_painel(caminho_saida, titulo, paineis):
    n = len(paineis)
    fig, eixos = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    if n == 1:
        eixos = [eixos]
    for ax, (titulo_p, base, contorno, cor) in zip(eixos, paineis):
        ax.imshow(base, cmap="gray")
        if contorno is not None and contorno.sum() > 0:
            ax.contour(contorno, levels=[0.5], colors=[cor], linewidths=2)
        ax.set_title(titulo_p, fontsize=11)
        ax.axis("off")
    fig.suptitle(titulo, fontsize=13)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    print("  Salvo: %s" % caminho_saida)


def processar_paciente(paciente, cluster_ids, params_classico, metricas_classico_df):
    pasta_paciente = os.path.join(CAMINHO_SAIDA, paciente)
    cam_volume = os.path.join(pasta_paciente, "volume_tc.nii.gz")
    if not os.path.exists(cam_volume):
        print("  [ERRO] volume não encontrado para %s — pulando" % paciente)
        return

    volume_hu = sitk.GetArrayFromImage(sitk.ReadImage(cam_volume))
    shape = volume_hu.shape

    cluster_id, foi_encontrado = escolher_nodulo(paciente, cluster_ids, metricas_classico_df)
    mascara_gt = _mod7.carregar_mascara_nodulo(pasta_paciente, cluster_id, shape)

    if mascara_gt.sum() == 0:
        print("  [ERRO] Máscara vazia para %s, cluster %d — pulando" % (
            paciente, cluster_id))
        return

    volume_norm_classico = normalizar_hu_classico(volume_hu)
    volume_norm_unet = _mod9.normalizar_hu(volume_hu)
    modelo = processar_paciente.modelo

    z, preds_classico_por_z, preds_unet_por_z = escolher_z_com_melhor_sobreposicao(
        mascara_gt, volume_norm_classico, volume_norm_unet, params_classico, modelo)
    if z is None:
        print("  [ERRO] Não foi possível escolher slice para %s — pulando" % paciente)
        return

    status = "detectado pelo clássico" if foi_encontrado else "NÃO detectado pelo clássico"
    print("  %s: nódulo (cluster %d, %s), slice z=%d (escolhido por "
          "sobreposição combinada máxima entre os dois métodos)" % (
              paciente, cluster_id, status, z))

    # Reaproveita as predições já calculadas para esse z, evitando rodar
    # os métodos de novo no mesmo slice.
    pred_classico = preds_classico_por_z[z]
    pred_unet = preds_unet_por_z[z]

    ct_slice = janela_hu(volume_hu[z])
    gt_slice = mascara_gt[z]

    sl_y, sl_x = bbox_com_folga(gt_slice, folga=25)
    ct_crop = ct_slice[sl_y, sl_x]
    gt_crop = gt_slice[sl_y, sl_x]
    classico_crop = pred_classico[sl_y, sl_x]
    unet_crop = pred_unet[sl_y, sl_x]

    os.makedirs(CAMINHO_SAIDA_FIGURAS, exist_ok=True)
    sufixo_status = "" if foi_encontrado else " (nódulo não detectado pelo clássico)"
    titulo_base = "%s — nódulo %d, slice %d%s" % (paciente, cluster_id, z, sufixo_status)

    salvar_painel(
        os.path.join(CAMINHO_SAIDA_FIGURAS, "%s_classico.png" % paciente),
        titulo_base,
        [
            ("CT", ct_crop, None, None),
            ("Referência", ct_crop, gt_crop, "lime"),
            ("Método clássico", ct_crop, classico_crop, "red"),
        ],
    )

    salvar_painel(
        os.path.join(CAMINHO_SAIDA_FIGURAS, "%s_unet.png" % paciente),
        titulo_base,
        [
            ("CT", ct_crop, None, None),
            ("Referência", ct_crop, gt_crop, "lime"),
            ("U-Net", ct_crop, unet_crop, "cyan"),
        ],
    )

    salvar_painel(
        os.path.join(CAMINHO_SAIDA_FIGURAS, "%s_comparacao.png" % paciente),
        titulo_base,
        [
            ("CT", ct_crop, None, None),
            ("Referência", ct_crop, gt_crop, "lime"),
            ("Método clássico", ct_crop, classico_crop, "red"),
            ("U-Net", ct_crop, unet_crop, "cyan"),
        ],
    )


def main():
    with open(os.path.join(CAMINHO_SLICES, "melhores_parametros.json")) as f:
        params_classico = json.load(f)
    print("Parâmetros do método clássico carregados:", params_classico)

    metricas_classico_df = pd.read_csv(CAMINHO_METRICAS_NODULO_CLASSICO)

    print("Carregando modelo U-Net (dispositivo: %s)..." % _mod9.DISPOSITIVO)
    processar_paciente.modelo = _mod9.carregar_modelo()

    nodulos_df = pd.read_csv(CAMINHO_NODULOS)
    clusters_por_paciente = (
        nodulos_df.groupby("paciente")["cluster_id"].unique().apply(list).to_dict()
    )

    print("\nGerando exemplos qualitativos para %d paciente(s)...\n" %
          len(PACIENTES_EXEMPLO))
    for paciente in PACIENTES_EXEMPLO:
        cluster_ids = clusters_por_paciente.get(paciente, [])
        if not cluster_ids:
            print("  [AVISO] %s sem nódulos válidos — pulando" % paciente)
            continue
        processar_paciente(paciente, cluster_ids, params_classico, metricas_classico_df)

    print("\nConcluído. Figuras salvas em: %s" % CAMINHO_SAIDA_FIGURAS)


if __name__ == "__main__":
    main()
