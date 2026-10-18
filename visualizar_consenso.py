# =============================================================================
# Título do Projeto: Comparação entre Método Clássico de Processamento Digital 
#                    de Imagens e Rede Neural para Segmentação de Nódulos 
#                    Pulmonares em Tomografias Computadorizadas
# Autores:           Daniel Augusto de Carvalho e Leonardo Reis dos Santos Melati
# Instituição:       UFABC (Universidade Federal do ABC)
# Ano:               2026
# Arquivo:           visualizar_consenso.py
# Descrição:         Script auxiliar para validação visual do agrupamento de 
#                    nódulos e comparação entre anotações e consensos.
# =============================================================================

import os
import re
import sys
import glob
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Janela de exibição em Hounsfield Units (padrão para tecido pulmonar/nódulo)
HU_MIN, HU_MAX = -1000, 400

# Cores para até 4 radiologistas distintos
CORES_RADIOLOGISTAS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231"]
COR_VOTACAO = "#ffe119"
COR_STAPLE  = "#911eb4"


def carregar_nifti(caminho):
    img = sitk.ReadImage(caminho)
    return sitk.GetArrayFromImage(img)


def janela_hu(slice_ct):
    """Normaliza um slice de CT para exibição em [0, 1] na janela HU definida."""
    clipped = np.clip(slice_ct, HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


def slice_central(mascara):
    """Retorna o índice do slice axial com maior área de máscara."""
    areas = mascara.sum(axis=(1, 2))
    if areas.max() == 0:
        return mascara.shape[0] // 2
    return int(np.argmax(areas))


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    soma = a.sum() + b.sum()
    if soma == 0:
        return 1.0
    return 2.0 * inter / soma


def bbox_com_folga(mascara, folga=15):
    coords = np.argwhere(mascara)
    if len(coords) == 0:
        return slice(0, mascara.shape[0]), slice(0, mascara.shape[1])
    y0, x0 = coords[:, 1].min(), coords[:, 2].min()
    y1, x1 = coords[:, 1].max(), coords[:, 2].max()
    H, W = mascara.shape[1], mascara.shape[2]
    y0 = max(0, y0 - folga); x0 = max(0, x0 - folga)
    y1 = min(H, y1 + folga); x1 = min(W, x1 + folga)
    return slice(y0, y1), slice(x0, x1)


def agrupar_arquivos_por_cluster(pasta_paciente):
    """
    Varre a pasta do paciente e agrupa os arquivos por cluster_id, a partir
    do padrão de nomes gerado pelo script 1 (mascara_noduloNN_rad*_annX,
    mascara_consenso_noduloNN_votacao/staple).
    """
    individuais = defaultdict(list)
    consenso = defaultdict(dict)

    for f in glob.glob(os.path.join(pasta_paciente, "mascara_nodulo*.nii.gz")):
        nome = os.path.basename(f)
        m = re.match(r"mascara_nodulo(\d+)_rad(.+)_ann(.+)\.nii\.gz", nome)
        if m:
            cluster_id = int(m.group(1))
            individuais[cluster_id].append({
                "radiologista": m.group(2),
                "nodulo_id": m.group(3),
                "arquivo": f,
            })

    for f in glob.glob(os.path.join(pasta_paciente, "mascara_consenso_nodulo*.nii.gz")):
        nome = os.path.basename(f)
        m = re.match(r"mascara_consenso_nodulo(\d+)_(votacao|staple)\.nii\.gz", nome)
        if m:
            cluster_id = int(m.group(1))
            consenso[cluster_id][m.group(2)] = f

    clusters_ids = sorted(set(individuais.keys()) | set(consenso.keys()))
    return clusters_ids, individuais, consenso


def visualizar_paciente(pasta_paciente, pasta_saida=None):
    nome_paciente = os.path.basename(pasta_paciente.rstrip("/\\"))
    cam_volume = os.path.join(pasta_paciente, "volume_tc.nii.gz")
    if not os.path.exists(cam_volume):
        print("[ERRO] volume_tc.nii.gz não encontrado em %s" % pasta_paciente)
        return

    volume = carregar_nifti(cam_volume)
    clusters_ids, individuais, consenso = agrupar_arquivos_por_cluster(pasta_paciente)

    if not clusters_ids:
        print("[AVISO] Nenhum nódulo encontrado para %s" % nome_paciente)
        arquivos = sorted(glob.glob(os.path.join(pasta_paciente, "mascara*.nii.gz")))
        if arquivos:
            print("  Arquivos 'mascara*' presentes na pasta (%d), mas nenhum "
                  "bateu com o padrão esperado pelo script de visualização:" %
                  len(arquivos))
            for a in arquivos[:15]:
                print("    - %s" % os.path.basename(a))
            if len(arquivos) > 15:
                print("    ... e mais %d arquivo(s)" % (len(arquivos) - 15))
            print("  -> Provavelmente esses arquivos foram gerados com uma "
                  "versão mais antiga do script 1 (nomes sem 'nodulo_rad_ann'). "
                  "Reprocesse este paciente com a versão mais recente do "
                  "script 1, ou me avise o padrão de nome acima para eu "
                  "ajustar o regex de leitura.")
        else:
            print("  Nenhum arquivo 'mascara*.nii.gz' encontrado na pasta — "
                  "confira se o caminho está correto.")
        return

    if pasta_saida is None:
        pasta_saida = os.path.join(pasta_paciente, "validacao")
    os.makedirs(pasta_saida, exist_ok=True)

    print("\n%s\nPaciente: %s | %d nódulo(s) físico(s)\n%s" % (
        "=" * 60, nome_paciente, len(clusters_ids), "=" * 60))

    for cluster_id in clusters_ids:
        anotacoes = individuais.get(cluster_id, [])
        masks_ind = [(a, carregar_nifti(a["arquivo"])) for a in anotacoes]

        mask_votacao = (carregar_nifti(consenso[cluster_id]["votacao"])
                         if "votacao" in consenso.get(cluster_id, {}) else None)
        mask_staple = (carregar_nifti(consenso[cluster_id]["staple"])
                        if "staple" in consenso.get(cluster_id, {}) else None)

        # Slice central: usa a máscara de votação se existir, senão a 1ª individual
        ref_mask = mask_votacao if mask_votacao is not None else (
            masks_ind[0][1] if masks_ind else None)
        if ref_mask is None:
            continue
        z = slice_central(ref_mask)
        sl_y, sl_x = bbox_com_folga(ref_mask, folga=20)

        ct_slice = janela_hu(volume[z])[sl_y, sl_x]

        fig, eixos = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("%s — Nódulo físico %02d (slice axial %d)" % (
            nome_paciente, cluster_id, z), fontsize=12)

        # Painel 1: anotações individuais
        ax = eixos[0]
        ax.imshow(ct_slice, cmap="gray")
        legendas = []
        for i, (info, m) in enumerate(masks_ind):
            cor = CORES_RADIOLOGISTAS[i % len(CORES_RADIOLOGISTAS)]
            contorno = m[z][sl_y, sl_x]
            if contorno.sum() > 0:
                ax.contour(contorno, levels=[0.5], colors=[cor], linewidths=1.5)
            legendas.append(mpatches.Patch(
                color=cor, label="Rad. %s" % info["radiologista"]))
        ax.set_title("Anotações individuais (n=%d)" % len(masks_ind))
        ax.legend(handles=legendas, fontsize=8, loc="lower right")
        ax.axis("off")

        # Painel 2: consenso votação
        ax = eixos[1]
        ax.imshow(ct_slice, cmap="gray")
        if mask_votacao is not None:
            contorno = mask_votacao[z][sl_y, sl_x]
            if contorno.sum() > 0:
                ax.contour(contorno, levels=[0.5], colors=[COR_VOTACAO], linewidths=2)
        ax.set_title("Consenso — Votação")
        ax.axis("off")

        # Painel 3: consenso STAPLE
        ax = eixos[2]
        ax.imshow(ct_slice, cmap="gray")
        if mask_staple is not None:
            contorno = mask_staple[z][sl_y, sl_x]
            if contorno.sum() > 0:
                ax.contour(contorno, levels=[0.5], colors=[COR_STAPLE], linewidths=2)
        ax.set_title("Consenso — STAPLE")
        ax.axis("off")

        plt.tight_layout()
        cam_fig = os.path.join(
            pasta_saida, "%s_nodulo%02d.png" % (nome_paciente, cluster_id))
        plt.savefig(cam_fig, dpi=130)
        plt.close(fig)

        # Resumo numérico no console
        n_ann = len(masks_ind)
        d = (dice(mask_votacao, mask_staple)
             if mask_votacao is not None and mask_staple is not None else None)
        print("  Nódulo %02d | %d anotações | vox votação=%s | vox STAPLE=%s | "
              "Dice(votação,STAPLE)=%s | figura: %s" % (
            cluster_id, n_ann,
            mask_votacao.sum() if mask_votacao is not None else "-",
            mask_staple.sum() if mask_staple is not None else "-",
            ("%.3f" % d) if d is not None else "-",
            os.path.basename(cam_fig)))

        if d is not None and d < 0.5:
            print("    [ATENÇÃO] Dice baixo entre votação e STAPLE — vale "
                  "abrir a figura e conferir visualmente esse nódulo.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python 2_visualizar_consenso.py <pasta_do_paciente> "
              "[<pasta_do_paciente_2> ...]")
        print("Exemplo: python 2_visualizar_consenso.py "
              "D:\\TG\\processado\\LIDC-IDRI-0078")
        sys.exit(1)

    for pasta in sys.argv[1:]:
        visualizar_paciente(pasta)

    print("\nConcluído. As figuras foram salvas em <pasta_paciente>/validacao/")
