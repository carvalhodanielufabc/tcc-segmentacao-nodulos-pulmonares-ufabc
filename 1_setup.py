# =============================================================================
# Título do Projeto: Comparação entre Método Clássico de Processamento Digital 
#                    de Imagens e Rede Neural para Segmentação de Nódulos 
#                    Pulmonares em Tomografias Computadorizadas
# Autores:           Daniel Augusto de Carvalho e Leonardo Reis dos Santos Melati
# Instituição:       UFABC (Universidade Federal do ABC)
# Ano:               2026
# Arquivo:           1_setup.py
# Descrição:         Instalação e verificação das dependências (bibliotecas) 
#                    necessárias para a execução do projeto.
# =============================================================================

import subprocess
import sys

PACOTES = [
    "pydicom",        # leitura de arquivos DICOM
    "SimpleITK",      # processamento de imagens médicas
    "numpy",          # operações numéricas
    "scipy",          # morfologia e operações científicas
    "scikit-image",   # segmentação clássica (thresholding, etc.)
    "matplotlib",     # visualização
    "pandas",         # organização dos resultados em tabelas
    "pyradiomics",    # extração de features radiômicas
]

print("=== Instalando dependências do TG ===\n")
for pacote in PACOTES:
    print(f"  Instalando {pacote}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pacote, "-q"])

print("\n=== Verificando importações ===\n")
nomes_import = {
    "pydicom":      "pydicom",
    "SimpleITK":    "SimpleITK",
    "numpy":        "numpy",
    "scipy":        "scipy",
    "scikit-image": "skimage",
    "matplotlib":   "matplotlib",
    "pandas":       "pandas",
    "pyradiomics":  "radiomics",
}

erros = []
for pacote, modulo in nomes_import.items():
    try:
        __import__(modulo)
        print(f"  [OK]   {pacote}")
    except ImportError:
        print(f"  [ERRO] {pacote}")
        erros.append(pacote)

if erros:
    print(f"\nATENÇÃO: falha ao importar: {erros}")
    print("Tente instalar manualmente: pip install " + " ".join(erros))
else:
    print("\nTudo instalado! Pode rodar o script 1_ler_dados.py")
