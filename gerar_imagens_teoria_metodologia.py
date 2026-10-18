# =============================================================================
# Título do Projeto: Comparação entre Método Clássico de Processamento Digital 
#                    de Imagens e Rede Neural para Segmentação de Nódulos 
#                    Pulmonares em Tomografias Computadorizadas
# Autores:           Daniel Augusto de Carvalho e Leonardo Reis dos Santos Melati
# Instituição:       UFABC (Universidade Federal do ABC)
# Ano:               2026
# Arquivo:           gerar_imagens_teoria_metodologia.py
# Descrição:         Script auxiliar para geração de figuras didáticas das 
#                    etapas de pré-processamento para o documento do TCC.
# =============================================================================

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage
from importlib.machinery import SourceFileLoader

# --- Configurações ---

CAMINHO_SAIDA = r"D:\TG\processadoV4"
CAMINHO_FIGURAS = r"D:\TG\figuras_teoria_metodologia"
PACIENTE_EXEMPLO = "LIDC-IDRI-0997"  # troque se preferir outro paciente

HU_MIN, HU_MAX = -1000, 400
LIMIAR_HU_DEMONSTRACAO = -400  # só para ilustrar a limiarização na 3.5.5

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_mod5 = SourceFileLoader(
    "metodo_classico", os.path.join(_SCRIPT_DIR, "5_metodo_classico_calibrar.py")
).load_module()


def janela_hu(slice_hu, vmin=HU_MIN, vmax=HU_MAX):
    clip = np.clip(slice_hu, vmin, vmax)
    return (clip - vmin) / (vmax - vmin)


def normalizar_hu(volume_hu):
    clip = np.clip(volume_hu, HU_MIN, HU_MAX)
    return ((clip - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def escolher_slice_pulmao_representativo(volume_norm):
    """Escolhe o slice com maior área pulmonar segmentada — tende a ser um
    corte 'central' do tórax, bom para ilustração."""
    areas = []
    for z in range(volume_norm.shape[0]):
        pulmao = _mod5.segmentar_pulmao(volume_norm[z])
        areas.append(pulmao.sum())
    return int(np.argmax(areas))


def gerar_figura_3_2(volume_hu, z, pasta_saida):
    slice_hu = volume_hu[z]
    img = janela_hu(slice_hu)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(slice_hu, cmap="gray", vmin=HU_MIN, vmax=HU_MAX)
    ax.set_title("Corte axial — janela de %d a %d HU" % (HU_MIN, HU_MAX))
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Unidades Hounsfield (HU)")
    plt.tight_layout()
    caminho = os.path.join(pasta_saida, "3.2_janela_hu.png")
    plt.savefig(caminho, dpi=150)
    plt.close(fig)
    print("Salvo: %s" % caminho)


def gerar_figura_3_5_5(volume_norm, pulmao, z, pasta_saida):
    limiar_norm = (LIMIAR_HU_DEMONSTRACAO - HU_MIN) / (HU_MAX - HU_MIN)
    bruto = (volume_norm[z] > limiar_norm) & pulmao

    estrutura = np.ones((3, 3), dtype=bool)
    aberta = ndimage.binary_opening(bruto, structure=estrutura)
    fechada = ndimage.binary_closing(aberta, structure=estrutura)

    # Recorte em torno da região pulmonar para não mostrar a imagem inteira
    coords = np.argwhere(pulmao)
    y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0)

    fig, eixos = plt.subplots(1, 3, figsize=(13, 4.6))
    paineis = [
        ("Limiarização bruta", bruto),
        ("Após abertura", aberta),
        ("Após abertura + fechamento", fechada),
    ]
    for ax, (titulo, mascara) in zip(eixos, paineis):
        ax.imshow(volume_norm[z][y0:y1, x0:x1], cmap="gray")
        ax.contour(mascara[y0:y1, x0:x1], levels=[0.5], colors=["red"], linewidths=1.2)
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    caminho = os.path.join(pasta_saida, "3.5.5_morfologia.png")
    plt.savefig(caminho, dpi=150)
    plt.close(fig)
    print("Salvo: %s" % caminho)
    print("  Nota: use um slice/paciente com candidatos visíveis (vasos, "
          "nódulo) para essa figura ficar didática. Se a região escolhida "
          "não tiver nada de interessante, troque PACIENTE_EXEMPLO ou o "
          "critério de escolha do slice.")


def gerar_figura_4_2(volume_hu, volume_norm, z, pulmao, pasta_saida):
    slice_hu = volume_hu[z]
    slice_clip = np.clip(slice_hu, HU_MIN, HU_MAX)
    slice_norm = volume_norm[z]

    fig, eixos = plt.subplots(1, 4, figsize=(16, 4.6))

    eixos[0].imshow(slice_hu, cmap="gray")  # janela automática (dinâmica completa)
    eixos[0].set_title("1. Slice bruto (HU completo)", fontsize=10)

    eixos[1].imshow(slice_clip, cmap="gray", vmin=HU_MIN, vmax=HU_MAX)
    eixos[1].set_title("2. Após clipping (%d a %d HU)" % (HU_MIN, HU_MAX), fontsize=10)

    eixos[2].imshow(slice_norm, cmap="gray", vmin=0, vmax=1)
    eixos[2].set_title("3. Após normalização (0 a 1)", fontsize=10)

    eixos[3].imshow(slice_norm, cmap="gray")
    eixos[3].contour(pulmao, levels=[0.5], colors=["lime"], linewidths=1.5)
    eixos[3].set_title("4. Segmentação pulmonar", fontsize=10)

    for ax in eixos:
        ax.axis("off")
    plt.tight_layout()
    caminho = os.path.join(pasta_saida, "4.2_preprocessamento.png")
    plt.savefig(caminho, dpi=150)
    plt.close(fig)
    print("Salvo: %s" % caminho)


def main():
    os.makedirs(CAMINHO_FIGURAS, exist_ok=True)

    pasta_paciente = os.path.join(CAMINHO_SAIDA, PACIENTE_EXEMPLO)
    cam_volume = os.path.join(pasta_paciente, "volume_tc.nii.gz")
    if not os.path.exists(cam_volume):
        print("[ERRO] Volume não encontrado: %s" % cam_volume)
        print("Ajuste PACIENTE_EXEMPLO para um paciente existente em %s" % CAMINHO_SAIDA)
        return

    volume_hu = sitk.GetArrayFromImage(sitk.ReadImage(cam_volume))
    volume_norm = normalizar_hu(volume_hu)

    print("Escolhendo slice representativo (maior área pulmonar)...")
    z = escolher_slice_pulmao_representativo(volume_norm)
    print("Paciente: %s | slice escolhido: z=%d\n" % (PACIENTE_EXEMPLO, z))

    pulmao = _mod5.segmentar_pulmao(volume_norm[z])

    gerar_figura_3_2(volume_hu, z, CAMINHO_FIGURAS)
    gerar_figura_3_5_5(volume_norm, pulmao, z, CAMINHO_FIGURAS)
    gerar_figura_4_2(volume_hu, volume_norm, z, pulmao, CAMINHO_FIGURAS)

    print("\nConcluído. Figuras salvas em: %s" % CAMINHO_FIGURAS)


if __name__ == "__main__":
    main()
