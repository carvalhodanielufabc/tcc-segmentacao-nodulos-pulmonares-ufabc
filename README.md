# Comparação entre Método Clássico de Processamento Digital de Imagens e Rede Neural para Segmentação de Nódulos Pulmonares em Tomografias Computadorizadas

## 👥 Autores e Orientação
* **Autores:** Daniel Augusto de Carvalho e Leonardo dos Santos Melati
* **Orientador:** Prof. Dr. Luiz Antonio Celiberto Junior
* **Instituição:** UFABC (Universidade Federal do ABC)
* **Ano:** 2026

---

## ⚠️ Aviso Importante: Configuração de Diretórios Locais
Este repositório mantém a estrutura original dos caminhos de diretórios utilizada no ambiente de desenvolvimento dos autores. **Antes de executar os scripts em sua máquina**, verifique e edite manualmente as variáveis de caminho (*paths*) no início de cada arquivo `.py` para apontar corretamente para a pasta raiz onde você salvou o seu conjunto de dados e onde deseja salvar os artefatos processados do projeto.

---

## 🛠️ Ambientes de Execução
Para garantir a reprodutibilidade dos experimentos descritos no TCC, o pipeline foi dividido em dois ambientes computacionais distintos, respeitando a exigência de processamento de cada etapa:

1. **Ambiente Local (Visual Studio Code + Python):** Utilizado para a preparação do banco de dados, filtragem, extração de fatias (*slices*), execução e calibração do método clássico de processamento de imagens, e geração de gráficos auxiliares. Requer ambiente Python com as bibliotecas descritas no script de setup.
2. **Ambiente em Nuvem (Google Colab com GPU NVIDIA T4):** Utilizado exclusivamente para o **Script 8 (`8_treinar_unet.py`)**, devido à alta demanda computacional e necessidade de aceleração por hardware (GPU) para otimizar o treinamento dos pesos da rede neural convolucional U-Net de forma eficiente.

---

## 📦 Dependências e Instalação
Para configurar o ambiente Python localmente, execute o script de setup fornecido ou instale manualmente as bibliotecas base:
`numpy`, `pandas`, `scikit-learn`, `scikit-image`, `opencv-python`, `pydicom`, `nibabel`, `pylidc`, `torch`, `torchvision`, `matplotlib`, `seaborn`.

---

## 📂 Estrutura e Descrição Detalhada do Pipeline (Scripts 0 a 11)

O pipeline experimental está modularizado em 12 etapas sequenciais que cobrem desde a extração dos dados brutos em formato DICOM até a avaliação estatística comparativa final.

### 1. Preparação e Pré-processamento de Dados
* **`0_script_selecionar_pacientes.py`**: Interage com a API da base pública LIDC-IDRI (via `pylidc`). Realiza varredura nos metadados clínicos para selecionar os 100 pacientes mais adequados, aplicando filtros rigorosos: exames com espessura de corte transversal <= 2.0 mm, presença de nódulos com diâmetro >= 3.0 mm e anotação validada por um mínimo de 2 radiologistas.
* **`1_setup.py`**: Script utilitário que automatiza a checagem, a instalação via `pip` e a validação de importação de todo o ecossistema de bibliotecas necessárias para rodar o pipeline, garantindo que o ambiente virtual esteja integro.
* **`2_ler_dados.py`**: Etapa crítica de conversão e padronização geométrica. Lê os arquivos DICOM brutos e as marcações XML. Converte os volumes para o formato padrão NIfTI (3D). Durante esse processo, aplica a janela radiológica de *Hounsfield Units* (HU) específica para dar contraste ao tecido pulmonar e resolve divergências entre especialistas utilizando critérios de concordância espacial (votação booleana / STAPLE) para gerar a máscara de consenso (Ground Truth).
* **`3_filtrar_e_dividir.py`**: Aplica a filtragem final das máscaras válidas e divide o *dataset* em subconjuntos de Treino, Validação e Teste. A divisão é estritamente **estratificada por paciente** — garantindo que fatias de um mesmo paciente não vazem entre os conjuntos (*data leakage*), o que comprometeria a validade científica da rede neural.
* **`4_gerar_slices_2d.py`**: Transforma a abordagem volumétrica 3D em um problema tratável em 2D. Extrai fatias axiais dos volumes NIfTI. Executa uma segmentação morfológica limiarizada do parênquima pulmonar (para ignorar ossos e ar externo) e recorta *patches* (janelas menores) focados em regiões com nódulos (amostras positivas) e áreas de tecido saudável (amostras negativas/background) para balancear o aprendizado.

### 2. Abordagem Clássica de Processamento Digital de Imagens
* **`5_metodo_classico_calibrar.py`**: Rotina de otimização de hiperparâmetros (semelhante a um *Grid Search*). Utiliza o conjunto de **Validação** para testar iterativamente diferentes combinações de limiares de intensidade (HU), filtros de área mínima/máxima de pixels conectados, e operações morfológicas (erosão/dilatação). O objetivo é encontrar a configuração heurística que maximize o acerto na segmentação.
* **`6_metodo_classico_avaliar_teste.py`**: Congela os hiperparâmetros obtidos no script anterior e os aplica de forma cega ao conjunto de **Teste**. Realiza a inferência em volumes 3D completos e computa métricas rigorosas de segmentação em nível de pixel: Índice Jaccard (IoU), Coeficiente Dice, Sensibilidade (*Recall*) e Especificidade.
* **`7_metodo_classico_avaliar_por_nodulo.py`**: Transita a avaliação do nível de pixel para o nível clínico/espacial (objeto 3D). Verifica o centróide das segmentações clássicas contra o Ground Truth para determinar taxas de acerto físico, contabilizando Verdadeiros Positivos (lesões detectadas), Falsos Negativos (lesões ignoradas) e Falsos Positivos (estruturas como vasos sanguíneos confundidas com nódulos).

### 3. Abordagem de Aprendizado Profundo (Deep Learning - U-Net)
* **`8_treinar_unet.py`** *(Executado no Google Colab com GPU NVIDIA T4)*: Constrói e treina a arquitetura convolucional U-Net (PyTorch). Utiliza técnica de *sliding window* (janela deslizante) para alimentar os *patches* 2D. Para mitigar o severo desbalanceamento de classes (poucos pixels de nódulo vs. milhares de pixels de fundo), a rede otimiza seus pesos utilizando uma função de perda especializada (*Focal Loss* ou *Dice Loss*).
* **`9_unet_calibrar_pos_processamento.py`**: Recebe as matrizes de probabilidade geradas pela U-Net no conjunto de Validação e aplica uma calibração fina. Define o ponto de corte ideal (*probability threshold*) e a área mínima de ruído a ser descartada, salvando essas regras matemáticas no arquivo de configuração `pos_processamento.json`.
* **`10_unet_avaliar_teste.py`**: Executa a inferência da U-Net sobre o conjunto de **Teste** ininterrupto. Aplica a técnica de *sliding window* dinâmica para juntar os recortes 2D de volta em um volume 3D coerente. Em seguida, aplica as regras do arquivo JSON para gerar a máscara binária predita e calcula as métricas de similaridade de pixel (Jaccard e Dice).
* **`11_unet_avaliar_por_nodulo.py`**: Semelhante ao Script 7, realiza a avaliação da U-Net no espaço volumétrico 3D para detecção de instâncias. Calcula as taxas de sensibilidade por nódulo físico e quantifica a incidência de Falsos Positivos, permitindo uma comparação pareada direta entre o desempenho da Inteligência Artificial e a heurística Clássica.

---

## 📈 Scripts Auxiliares e de Apoio Documental
* **`visualizar_consenso.py`**: Ferramenta visual de *debugging* técnico. Plota mapas de calor e *overlays* para evidenciar o grau de sobreposição e discordância entre os diferentes médicos radiologistas da base LIDC-IDRI em relação à máscara de consenso final.
* **`gerar_imagens_teoria_metodologia.py`**: Script voltado puramente para apoio acadêmico. Renderiza gráficos, histogramas de distribuição de Hounsfield Units e estágios passo a passo da segmentação do pulmão para ilustrar o documento escrito da monografia.
* **`gerar_exemplos_qualitativos.py`**: Seleciona e processa recortes representativos do Teste, gerando painéis visuais (*subplots* lado a lado) que comparam o Ground Truth, a predição Clássica e a predição da U-Net, fundamentais para a discussão qualitativa dos resultados no TCC.
