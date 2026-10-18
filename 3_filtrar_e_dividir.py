# =============================================================================
# Título do Projeto: Comparação entre Método Clássico de Processamento Digital 
#                    de Imagens e Rede Neural para Segmentação de Nódulos 
#                    Pulmonares em Tomografias Computadorizadas
# Autores:           Daniel Augusto de Carvalho e Leonardo Reis dos Santos Melati
# Instituição:       UFABC (Universidade Federal do ABC)
# Ano:               2026
# Arquivo:           3_filtrar_e_dividir.py
# Descrição:         Filtragem de nódulos válidos e divisão estratificada dos 
#                    pacientes nos conjuntos de Treino, Validação e Teste.
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configurações ---

CAMINHO_SAIDA = r"D:\TG\processadoV4"
CAMINHO_CSV_ENTRADA = os.path.join(CAMINHO_SAIDA, "indice_nodulos.csv")

# Nº mínimo de anotações (radiologistas) concordando para o cluster ser
# considerado um nódulo válido para treino/avaliação.
LIMIAR_ANOTACOES = 3

# Tamanho do conjunto de teste (mantido intocado até a comparação final).
# Com 100 pacientes: 70 treino / 10 validação / 20 teste.
N_TESTE = 20

# Tamanho do conjunto de validação, tirado do restante (usado para
# early stopping / ajuste de hiperparâmetros da U-Net).
N_VALIDACAO = 10
# O que sobrar (100 - N_TESTE - N_VALIDACAO = 70) vai para treino.

SEED = 7  # MANTIDO igual à V4 — a divisão automática abaixo é idêntica
          # à da V4; só a troca manual (FORCAR_PARA_TREINO) é nova.

# --- Troca manual de pacientes (NOVO na V5) ---
# Pacientes nesta lista são FORÇADOS para o treino, mesmo que a divisão
# automática (estratificada, com SEED acima) os tivesse colocado em
# validação ou teste. Para cada paciente movido, um substituto é
# automaticamente promovido do treino para o split de origem dele (ver
# aplicar_troca_manual), mantendo o tamanho de treino/val/teste igual ao
# da V4. Motivo de cada entrada documentado no cabeçalho do arquivo.
FORCAR_PARA_TREINO = ["LIDC-IDRI-0921"]

# --- Modo incremental ---
# Se True e já existir um split_pacientes.csv de uma rodada anterior,
# os pacientes que já tinham split definido MANTÊM o mesmo split
# (train/val/test) — o teste e a validação ficam intocados, protegendo
# a validade de tudo que já foi calibrado/avaliado com eles. Pacientes
# NOVOS (que não estavam no split anterior) são todos adicionados ao
# TREINO. Se False, ignora qualquer split anterior e sorteia do zero
# (só use False se quiser descartar a divisão anterior de propósito).
#
# Desligado agora de propósito: você optou por uma divisão nova do zero
# com os 100 pacientes (70/10/20), não por preservar o split anterior.
MODO_INCREMENTAL = False


def dividir_pacientes_incremental(resumo, cam_split_existente):
    split_existente = pd.read_csv(cam_split_existente)
    mapa_split = dict(zip(split_existente["paciente"], split_existente["split"]))

    resumo = resumo.copy()
    resumo["split"] = resumo["paciente"].map(mapa_split)

    novos = resumo["split"].isna()
    n_novos = int(novos.sum())
    n_existentes = len(resumo) - n_novos
    resumo.loc[novos, "split"] = "train"

    print("Modo incremental: %d paciente(s) mantiveram o split anterior, "
          "%d paciente(s) novo(s) adicionado(s) ao TREINO" % (n_existentes, n_novos))
    if n_novos > 0:
        print("  Novos pacientes: %s" % sorted(resumo.loc[novos, "paciente"].tolist()))

    pacientes_atuais = set(resumo["paciente"])
    pacientes_sumidos = set(mapa_split.keys()) - pacientes_atuais
    if pacientes_sumidos:
        print("  [AVISO] %d paciente(s) do split anterior não aparecem mais "
              "no indice_nodulos.csv atual (removidos ou não reprocessados): %s"
              % (len(pacientes_sumidos), sorted(pacientes_sumidos)))

    return resumo


def carregar_e_filtrar(caminho_csv, limiar_anotacoes):
    df = pd.read_csv(caminho_csv)

    obrigatorias = {"paciente", "cluster_id", "n_anotacoes_cluster"}
    faltando = obrigatorias - set(df.columns)
    if faltando:
        raise ValueError(
            "CSV não tem as colunas esperadas (%s). Confira se foi gerado "
            "pela versão mais recente do script 1." % faltando)

    # 1 linha por cluster (nódulo físico), independente do formato
    # XML ou DICOM-SEG — n_anotacoes_cluster já conta quantas anotações
    # formam aquele cluster, então serve como critério único para os dois.
    clusters = (
        df.groupby(["paciente", "cluster_id"], as_index=False)
        .agg(
            n_anotacoes_cluster=("n_anotacoes_cluster", "first"),
            malignidade_media=("malignidade", "mean"),
            volume_mm3_medio=("volume_mm3", "mean"),
        )
    )

    clusters["valido"] = clusters["n_anotacoes_cluster"] >= limiar_anotacoes

    n_total = len(clusters)
    n_validos = clusters["valido"].sum()
    print("Nódulos físicos (clusters) no total: %d" % n_total)
    print("Nódulos válidos (>= %d anotações): %d (%.1f%%)" % (
        limiar_anotacoes, n_validos, 100 * n_validos / n_total))

    return df, clusters


def resumo_por_paciente(clusters):
    resumo = (
        clusters.groupby("paciente")
        .agg(
            n_nodulos_total=("cluster_id", "count"),
            n_nodulos_validos=("valido", "sum"),
        )
        .reset_index()
    )

    sem_nodulo_valido = resumo[resumo["n_nodulos_validos"] == 0]
    if len(sem_nodulo_valido) > 0:
        print("\n[AVISO] %d paciente(s) ficaram SEM nenhum nódulo válido "
              "após o filtro (ainda são mantidos na divisão, mas não "
              "contribuem com nódulos positivos — úteis só como fonte de "
              "slices negativos, se for esse o seu desenho de treino):" %
              len(sem_nodulo_valido))
        for p in sem_nodulo_valido["paciente"]:
            print("    - %s" % p)

    return resumo


ORDEM_FAIXAS = ["0", "1", "2", "3+"]


def _mesclar_estratos_raros(estratos, ordem=ORDEM_FAIXAS, minimo=2):
    """
    train_test_split(stratify=...) exige pelo menos 2 membros por classe.
    Com poucos pacientes numa faixa (ex.: só 1 paciente sem nenhum
    nódulo válido), a estratificação quebra. Aqui mesclamos
    iterativamente qualquer classe com menos de `minimo` membros na
    classe vizinha (na ordem definida), até todas as classes restantes
    terem membros suficientes.
    """
    estratos = np.array(estratos, dtype=object)
    while True:
        valores, contagens = np.unique(estratos, return_counts=True)
        mapa_contagem = dict(zip(valores, contagens))
        pequenas = [v for v in ordem if v in mapa_contagem and mapa_contagem[v] < minimo]
        if not pequenas or len(mapa_contagem) <= 1:
            break
        classe = pequenas[0]
        idx = ordem.index(classe)
        alvo = None
        for j in range(idx + 1, len(ordem)):
            if ordem[j] in mapa_contagem:
                alvo = ordem[j]
                break
        if alvo is None:
            for j in range(idx - 1, -1, -1):
                if ordem[j] in mapa_contagem:
                    alvo = ordem[j]
                    break
        if alvo is None:
            break
        print("  [Estratificação] mesclando classe '%s' (%d paciente(s)) "
              "com '%s' — poucos membros para estratificar sozinha" % (
                  classe, mapa_contagem[classe], alvo))
        estratos[estratos == classe] = alvo
    return estratos


def dividir_pacientes(resumo, n_teste, n_validacao, seed):
    n_total = len(resumo)

    # Pacientes com ZERO nódulos válidos nunca vão para teste/validação
    # — não têm nenhum nódulo positivo para avaliar, então não servem
    # pra medir desempenho de detecção/segmentação nesses conjuntos.
    # Vão automaticamente para o treino (ainda são úteis lá como fonte
    # de slices negativos).
    mascara_zero = resumo["n_nodulos_validos"] == 0
    resumo_elegivel = resumo[~mascara_zero].reset_index(drop=True)
    resumo_forcado_treino = resumo[mascara_zero].copy()
    resumo_forcado_treino["split"] = "train"

    if len(resumo_forcado_treino) > 0:
        print("%d paciente(s) com ZERO nódulos válidos — excluídos de "
              "teste/validação, forçados para o treino: %s" % (
                  len(resumo_forcado_treino),
                  sorted(resumo_forcado_treino["paciente"].tolist())))

    n_elegivel = len(resumo_elegivel)
    n_treino_elegivel = n_elegivel - n_teste - n_validacao
    if n_treino_elegivel <= 0:
        raise ValueError(
            "N_TESTE + N_VALIDACAO (%d) >= pacientes elegíveis com nódulo "
            "válido (%d). Reduza N_TESTE/N_VALIDACAO ou revise "
            "LIMIAR_ANOTACOES." % (n_teste + n_validacao, n_elegivel))

    # Estratifica pela quantidade de nódulos válidos por paciente (em
    # faixas), para treino/val/teste ficarem comparáveis em dificuldade
    # — só entre os pacientes elegíveis (com >=1 nódulo válido).
    faixas = pd.cut(
        resumo_elegivel["n_nodulos_validos"],
        bins=[-1, 0, 1, 2, np.inf],
        labels=ORDEM_FAIXAS,
    )
    y_estrato = _mesclar_estratos_raros(faixas.astype(str).values)

    # Passo 1: separar teste do restante
    idx_resto, idx_teste = train_test_split(
        np.arange(n_elegivel),
        test_size=n_teste,
        random_state=seed,
        stratify=y_estrato,
    )

    # Passo 2: dentro do resto, separar validação do treino — remescla
    # as classes raras de novo, já que a distribuição mudou depois de
    # remover os pacientes de teste.
    y_resto = _mesclar_estratos_raros(y_estrato[idx_resto])
    idx_treino_rel, idx_val_rel = train_test_split(
        np.arange(len(idx_resto)),
        test_size=n_validacao,
        random_state=seed,
        stratify=y_resto,
    )
    idx_treino = idx_resto[idx_treino_rel]
    idx_val = idx_resto[idx_val_rel]

    split = pd.Series(index=np.arange(n_elegivel), dtype=object)
    split.iloc[idx_treino] = "train"
    split.iloc[idx_val] = "val"
    split.iloc[idx_teste] = "test"

    resumo_elegivel = resumo_elegivel.copy()
    resumo_elegivel["split"] = split.values

    resumo_final = pd.concat(
        [resumo_elegivel, resumo_forcado_treino], ignore_index=True)
    return resumo_final


def aplicar_troca_manual(resumo, forcar_para_treino, seed):
    """
    NOVO NA V5. Roda DEPOIS da divisão automática (dividir_pacientes),
    que é idêntica à da V4 (mesmo SEED). Para cada paciente em
    `forcar_para_treino`:

      1) Move o paciente para o treino, qualquer que seja o split onde
         a divisão automática o colocou.
      2) Promove automaticamente um paciente do treino para o split de
         origem (val ou test), para o tamanho de cada split continuar
         igual ao da V4 — senão o teste ficaria com 19 pacientes em vez
         de 20, por exemplo, quebrando a comparabilidade com a V4.

    O substituto é escolhido entre os pacientes hoje no treino com
    n_nodulos_validos mais PRÓXIMO do paciente removido (mesma faixa de
    dificuldade) — não o "melhor", porque não há como saber o
    desempenho de um paciente que nunca foi avaliado sem antes treinar
    e testar com ele. Em caso de empate, o desempate é determinístico
    (mesmo SEED da divisão), para o resultado ser reprodutível.
    """
    resumo = resumo.copy()
    ja_usados = set(forcar_para_treino)  # nunca promove um paciente que
                                          # também está sendo removido

    for paciente in forcar_para_treino:
        if paciente not in resumo["paciente"].values:
            print("  [AVISO] Paciente %s (em FORCAR_PARA_TREINO) não "
                  "encontrado no indice_nodulos.csv atual — ignorando." %
                  paciente)
            continue

        idx_paciente = resumo.index[resumo["paciente"] == paciente][0]
        split_origem = resumo.loc[idx_paciente, "split"]
        if split_origem == "train":
            print("  Paciente %s já está no treino — nada a trocar." %
                  paciente)
            continue

        n_nodulos_removido = resumo.loc[idx_paciente, "n_nodulos_validos"]
        resumo.loc[idx_paciente, "split"] = "train"

        candidatos = resumo[
            (resumo["split"] == "train") &
            (resumo["n_nodulos_validos"] > 0) &
            (~resumo["paciente"].isin(ja_usados))
        ].copy()
        if len(candidatos) == 0:
            print("  [AVISO] Nenhum paciente elegível no treino para "
                  "promover no lugar de %s — o split '%s' ficou com 1 "
                  "paciente a menos que a V4." % (paciente, split_origem))
            continue

        # Mais próximo em nº de nódulos válidos = mesma faixa de
        # dificuldade do paciente removido; desempate determinístico
        # (ordenado por nome do paciente, estável e reprodutível).
        candidatos["dist_nodulos"] = (
            candidatos["n_nodulos_validos"] - n_nodulos_removido).abs()
        candidatos = candidatos.sort_values(
            ["dist_nodulos", "paciente"], kind="stable")
        escolhido = candidatos.iloc[0]["paciente"]

        idx_escolhido = resumo.index[resumo["paciente"] == escolhido][0]
        resumo.loc[idx_escolhido, "split"] = split_origem
        ja_usados.add(escolhido)

        print("  Troca manual: %s (%d nódulo(s) válido(s), estava em "
              "'%s') -> TREINO  |  %s (%d nódulo(s) válido(s), estava "
              "no treino) -> '%s'" % (
                  paciente, n_nodulos_removido, split_origem,
                  escolhido, resumo.loc[idx_escolhido, "n_nodulos_validos"],
                  split_origem))
        print("  [ATENÇÃO] Não há garantia de que %s vai ter resultado "
              "melhor que %s — isso só se confirma depois de treinar e "
              "avaliar de novo. A escolha só preserva a dificuldade do "
              "split (nº de nódulos válidos parecido)." % (
                  escolhido, paciente))

    return resumo


def imprimir_resumo_split(resumo):
    print("\n" + "=" * 60)
    print("Divisão final (%d pacientes)" % len(resumo))
    print("=" * 60)
    for grupo in ["train", "val", "test"]:
        sub = resumo[resumo["split"] == grupo]
        print("  %-5s: %2d pacientes | nódulos válidos: total=%3d, "
              "média=%.2f/paciente" % (
                  grupo, len(sub), sub["n_nodulos_validos"].sum(),
                  sub["n_nodulos_validos"].mean()))


def main():
    df, clusters = carregar_e_filtrar(CAMINHO_CSV_ENTRADA, LIMIAR_ANOTACOES)
    resumo = resumo_por_paciente(clusters)

    cam_split = os.path.join(CAMINHO_SAIDA, "split_pacientes_V5.csv")
    if MODO_INCREMENTAL and os.path.exists(cam_split):
        resumo = dividir_pacientes_incremental(resumo, cam_split)
    else:
        if MODO_INCREMENTAL:
            print("MODO_INCREMENTAL=True, mas não há split_pacientes_V5.csv "
                  "anterior — fazendo a divisão do zero.")
        resumo = dividir_pacientes(resumo, N_TESTE, N_VALIDACAO, SEED)

    if FORCAR_PARA_TREINO:
        print("\n" + "=" * 60)
        print("Troca manual de pacientes (V5) — FORCAR_PARA_TREINO=%s" %
              FORCAR_PARA_TREINO)
        print("=" * 60)
        resumo = aplicar_troca_manual(resumo, FORCAR_PARA_TREINO, SEED)

    imprimir_resumo_split(resumo)

    # Salva o resumo por paciente (com o split)
    cam_split = os.path.join(CAMINHO_SAIDA, "split_pacientes_V5.csv")
    resumo.to_csv(cam_split, index=False)
    print("\nSalvo: %s" % cam_split)

    # Salva o CSV de nódulos, filtrado (só clusters válidos) e já com
    # a coluna 'split' — este é o arquivo que deve alimentar o treino
    # e a avaliação dos dois métodos de segmentação.
    clusters_validos = clusters[clusters["valido"]][["paciente", "cluster_id"]]
    df_validos = df.merge(clusters_validos, on=["paciente", "cluster_id"], how="inner")
    df_validos = df_validos.merge(
        resumo[["paciente", "split"]], on="paciente", how="left")

    cam_nodulos = os.path.join(CAMINHO_SAIDA, "nodulos_validos_V5.csv")
    df_validos.to_csv(cam_nodulos, index=False)
    print("Salvo: %s (%d linhas, de %d originais)" % (
        cam_nodulos, len(df_validos), len(df)))


if __name__ == "__main__":
    main()
