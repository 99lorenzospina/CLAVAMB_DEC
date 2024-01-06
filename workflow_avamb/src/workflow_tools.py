import numpy as np
import os
import json
from typing import cast, Optional
import vamb
import shutil


"""
La prima funzione per ricavare i punteggi di completeness e contamination per ciascun bin
La seconda funzione per aggiornarli dal dizionario
"""

def get_cluster_score_bin_path(
    path_checkm_all: str, path_bins: str, bins: set[str]
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    """Given CheckM has been run for all samples, create 2 dictionaries:
    - {bin:path_bin}
    - {bin:[completeness, contamination]}"""
    cluster_score: dict[str, tuple[float, float]] = dict()
    bin_path: dict[str, str] = dict()
    # Itera su tutte le cartelle (samples) in path_checkm_all
    for sample in os.listdir(path_checkm_all):

        # Costruisci il percorso completo al file quality_report.tsv
        path_quality_s = os.path.join(path_checkm_all, sample, "quality_report.tsv")

        # Carica il file quality_report.tsv come un array NumPy
        c_com_con = np.loadtxt(
            path_quality_s,
            delimiter="\t",
            skiprows=1,
            usecols=(0, 1, 2),
            dtype=str,
            ndmin=2,
        )

        # Itera su ogni riga nel file quality_report.tsv
        for row in c_com_con:

            # Estrai le informazioni dalla riga
            cluster, com, con = row
            cluster = cast(str, cluster)
            com, con = float(com), float(con)

            # Costruisci il nome del bin aggiungendo l'estensione .fna
            bin_name = cluster + ".fna"

            # Verifica se il bin è presente nella lista specificata
            if bin_name in bins:

                # Aggiungi le informazioni del cluster al dizionario cluster_score
                cluster_score[cluster] = (com, con)

                # Costruisci il percorso completo al bin e aggiungilo al dizionario bin_path
                bin_path[cluster + ".fna"] = os.path.join(
                    path_bins, sample, cluster + ".fna"
                )

    # Restituisci una tupla contenente i dizionari cluster_score e bin_path
    return cluster_score, bin_path


def update_cluster_score_bin_path(
    path_checkm_ripped: str, cluster_score: dict[str, tuple[float, float]]
) -> dict[str, tuple[float,float]] :
    c_com_con = np.loadtxt(
        path_checkm_ripped,
        delimiter="\t",
        skiprows=1,
        usecols=(0, 1, 2),
        dtype=str,
        ndmin=2,
    )
    
    # Itera su ogni riga nel file CheckM
    for row in c_com_con:
        cluster, com, con = row
        if "--" in cluster:
            continue
        com, con = float(com), float(con)
        print(cluster, "scores were", cluster_score[cluster])

        # Aggiorna i punteggi di qualità nel dizionario cluster_score
        cluster_score[cluster] = (com, con)
        print("and now are", cluster_score[cluster])

    # Restituisce il dizionario cluster_score aggiornato
    return cluster_score
