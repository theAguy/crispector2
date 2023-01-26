import pandas as pd
import numpy as np
import copy
import statistics
from typing import Dict
from algorithm.binomial_probability import compute_binom_p
from algorithm.core_algorithm import CoreAlgorithm
from utils.constants_and_types import FREQ, TX_READ_NUM, MOCK_READ_NUM, ON_TARGET, CUT_SITE, AlgResult
from modifications.modification_tables import ModificationTables
from modifications.modification_types import ModificationTypes
from input_processing.alignment import Alignment
from utils.configurator import Configurator


def create_random_reads_df_per_site(tables_data, re_run_sites):
    """
    Take all sites that need to be re-compute and aggregate all alleles' random reads into a separate dfs
    :param tables_data: tables_d. all the modifications of all sites, including the mock & tx dfs in it
    :param re_run_sites: all allele sites that has to be re-computed to determine best editing activity
    :return: the tables_d without "random_reads" for each site, a mock and treatment dfs of "random_reads" for each site
             and a map between sites to their alleles
    """
    random_mock = dict()
    random_tx = dict()
    site_to_alleles_link = dict()

    for site, alleles in re_run_sites.items():
        for allele, allele_info in alleles.items():
            allele_name = allele_info[0]
            # assign in site_to_alleles_link dictionary
            if site in site_to_alleles_link.keys():
                site_to_alleles_link[site].append(allele_name)
            else:
                site_to_alleles_link[site] = [allele_name]

            try:
                # take the df
                tx_df = tables_data[allele_name].tx_reads
                # filter the random reads
                random_tx_df = tx_df[tx_df['is_random'] == True]
                # assign to the original site. combine with other alleles random reads
                if site in random_tx.keys():
                    concat_df = pd.concat([random_tx[site], random_tx_df])
                    random_tx[site] = concat_df
                else:
                    random_tx[site] = random_tx_df
                # remove from the current df
                random_index = tx_df[tx_df['is_random'] == True].index
                tables_data[allele_name].tx_reads.drop(random_index, inplace=True)

            # if there are no random reads - pass this stage
            except:
                pass

            try:
                # take the df
                mock_df = tables_data[allele_name].mock_reads
                # filter the random reads
                random_mock_df = mock_df[mock_df['is_random'] == True]
                # assign to the original site. combine with other alleles random reads
                if site in random_mock.keys():
                    concat_df = pd.concat([random_mock[site], random_mock_df])
                    random_mock[site] = concat_df
                else:
                    random_mock[site] = random_mock_df
                # remove from the current df
                random_index = mock_df[mock_df['is_random'] == True].index
                tables_data[allele_name].mock_reads.drop(random_index, inplace=True)

            # if there are no random reads - pass this stage
            except:
                pass

    return tables_data, random_mock, random_tx, site_to_alleles_link


def random_assign_random_reads_to_dfs(tables_d, random_tx_dict, random_mock_dict, link_site_allele):
    """
    Randomly assign "random reads" to relevant alleles
    :param tables_d: original tables, without "random reads" in their dfs
    :param random_tx_dict: per each site, all the tx "random reads"
    :param random_mock_dict: per each site, all the mock "random reads"
    :param link_site_allele: the mapping between site to alleles
    :return: new tables_d with new random assignment of "random reads"
    """

    new_tables_d = copy.deepcopy(tables_d)

    for site, df in random_mock_dict.items():
        if len(df) != 0:
            num_of_alleles = len(link_site_allele[site])
            shuffled_random_df = df.sample(frac=1)
            shuffled_fraction_df = np.array_split(shuffled_random_df, num_of_alleles)

            for i in range(num_of_alleles):
                allele = link_site_allele[site][i]
                allele_mock_df = new_tables_d[allele].mock_reads
                random_df_to_add = shuffled_fraction_df[i]
                new_mock_df = pd.concat([allele_mock_df, random_df_to_add])

                new_tables_d[allele].mock_reads = new_mock_df

    for site, df in random_tx_dict.items():
        if len(df) != 0:
            num_of_alleles = len(link_site_allele[site])
            shuffled_random_df = df.sample(frac=1)
            shuffled_fraction_df = np.array_split(shuffled_random_df, num_of_alleles)

            for i in range(num_of_alleles):
                allele = link_site_allele[site][i]
                allele_tx_df = new_tables_d[allele].tx_reads
                random_df_to_add = shuffled_fraction_df[i]
                new_tx_df = pd.concat([allele_tx_df, random_df_to_add])

                new_tables_d[allele].tx_reads = new_tx_df

    return new_tables_d


def compute_best_stats(enable_substitutions, confidence_interval, donor, min_num_of_reads, override_noise_estimation,
                       allele_ref_df, dfs_data, re_run_overlapping_sites, amplicon_min_score,
                       translocation_amplicon_min_score):
    """
    Takes all ata regarding alleles sites (random and not random reads) and compute new statistics
    :param enable_substitutions: Flag
    :param confidence_interval: the confidence interval parameter
    :param donor: is donor experiment flag
    :param min_num_of_reads: minimum number of reads per site. hyperparameter
    :param override_noise_estimation: _
    :param allele_ref_df: the reference df with reference alleles
    :param dfs_data: tables_d. all the modifications of all sites, including the mock & tx dfs in it
    :param re_run_overlapping_sites: all allele sites that has to be re-computed to determine best editing activity
    :param amplicon_min_score: for re-alignment
    :param translocation_amplicon_min_score: for re-alignment
    :return: The new dfs of the alleles and the new statistics scores
    """

    # create tx_reads_d and mock_reads_d
    for site, alleles in re_run_overlapping_sites.items():
        allele_sites = list()
        for allele_name, allele_info in alleles.items():
            allele_sites.append(allele_info[0])

        tx_reads_d = dict()
        mock_reads_d = dict()

        for allele in allele_sites:
            tx_reads_d[allele] = dfs_data[allele].tx_reads.reset_index()
            mock_reads_d[allele] = dfs_data[allele].mock_reads.reset_index()

    # re align tx reads and mock - [TBD]: check if needed, maybe over-kill (it takes time)
    # tx_reads_d_align = re_align_df(tx_reads_d, allele_ref_df, amplicon_min_score, translocation_amplicon_min_score)
    # mock_reads_d_align = re_align_df(mock_reads_d, allele_ref_df, amplicon_min_score, translocation_amplicon_min_score)

    # Get modification types and positions priors
    modifications_allele = ModificationTypes.init_from_cfg(enable_substitutions)

    # Convert alignment to modification tables
    tables_d_allele: Dict[str, ModificationTables] = dict()
    for allele, row in allele_ref_df.iterrows():
        if allele in allele_sites:
            tx_reads_num = tx_reads_d[allele][FREQ].sum().astype(int)
            mock_reads_num = mock_reads_d[allele][FREQ].sum().astype(int)
            if donor and row[ON_TARGET]:
                pass
            elif min(tx_reads_num, mock_reads_num) < min_num_of_reads:
                pass
            else:
                tables_d_allele[allele] = ModificationTables(tx_reads_d[allele], mock_reads_d[allele],
                                                             modifications_allele, row)

    # Compute binomial coin for all modification types
    binom_p_d = compute_binom_p(tables_d_allele, modifications_allele, override_noise_estimation, allele_ref_df)
    # TBD: delete from binom the log
    # Run crispector core algorithm on all sites
    result_summary_d_allele: AlgResult = dict()  # Algorithm result dictionary
    algorithm_d: Dict[str, CoreAlgorithm] = dict()
    for allele, row in allele_ref_df.iterrows():
        if allele in allele_sites:
            cut_site = row[CUT_SITE]
            # Continue if site was discarded
            if allele not in tables_d_allele:
                # Log the following in the result dict
                tx_reads_num = tx_reads_d[allele][FREQ].sum().astype(int)
                mock_reads_num = mock_reads_d[allele][FREQ].sum().astype(int)
                result_summary_d_allele[allele] = {TX_READ_NUM: tx_reads_num, MOCK_READ_NUM: mock_reads_num,
                                                   ON_TARGET: row[ON_TARGET]}
                continue

            algorithm_d[allele] = CoreAlgorithm(cut_site, modifications_allele, binom_p_d[allele],
                                                confidence_interval, row[ON_TARGET])
            result_summary_d_allele[allele] = algorithm_d[allele].evaluate(tables_d_allele[allele])

    return tables_d_allele, result_summary_d_allele


def re_align_df(reads_df, ref_df, amplicon_min_score, translocation_amplicon_min_score):
    """
    Re-align df to the relevant amplicon
    :param reads_df:
    :param ref_df:
    :param amplicon_min_score: for re-alignment
    :param translocation_amplicon_min_score: for re-alignment
    :return: Re-aligned df
    """
    # set configuration and aligner for the alignment
    _cfg = Configurator.get_cfg()
    _aligner = Alignment(_cfg["alignment"], amplicon_min_score, translocation_amplicon_min_score,
                         _cfg["NHEJ_inference"]["window_size"])
    new_sites = reads_df.copy()

    # iterate over each new allele and re-align it
    for site, df in reads_df.items():
        cut_site = ref_df.loc[site, 'cut-site']
        ref_amplicon = ref_df.loc[site, 'AmpliconReference']

        new_allele_df = _aligner.align_reads(df, ref_amplicon, cut_site,
                                             None, None, site, None, allele=True)
        new_sites[site] = new_allele_df

    return new_sites


def get_best_random_reads_assignment(enable_substitutions, confidence_interval, donor, min_num_of_reads,
                                     override_noise_estimation, allele_ref_df, dfs_data, re_run_overlapping_sites,
                                     amplicon_min_score, translocation_amplicon_min_score):
    """
    function that takes all ambiguous sites (regarding CI), re-assign the random reads to the alleles and re-compute
    the statistics. This done 11 times, taking the median score of each site as the "best" score
    :param enable_substitutions: Flag
    :param confidence_interval: the confidence interval parameter
    :param donor: is donor experiment flag
    :param min_num_of_reads: minimum number of reads per site. hyperparameter
    :param override_noise_estimation: _
    :param allele_ref_df: the reference df with reference alleles
    :param dfs_data: tables_d. all the modifications of all sites, including the mock & tx dfs in it
    :param re_run_overlapping_sites: all allele sites that has to be re-computed to determine best editing activity
    :param amplicon_min_score: for re-alignment
    :param translocation_amplicon_min_score: for re-alignment
    :return: The best selected statistics and dfs out of 11 random possibles
    """
    # take out from mock and treatment dfs all the "random reads" and store them separately
    random_tables_d, random_mock_dict, random_tx_dict, link_site_allele = \
        create_random_reads_df_per_site(dfs_data, re_run_overlapping_sites)

    tables_d_dict = list()
    result_summary_d_dict = list()
    for i in range(11):
        # assign "random reads" randomly between sites
        tables_d_random = random_assign_random_reads_to_dfs(random_tables_d, random_tx_dict, random_mock_dict,
                                                            link_site_allele)

        # compute new statistics per each site that was modified randomly
        tables_d_allele, result_summary_d_allele = compute_best_stats(enable_substitutions, confidence_interval, donor,
                                                                      min_num_of_reads, override_noise_estimation,
                                                                      allele_ref_df, tables_d_random,
                                                                      re_run_overlapping_sites, amplicon_min_score,
                                                                      translocation_amplicon_min_score)
        # store the dfs and the statistics results
        tables_d_dict.append(tables_d_allele)
        result_summary_d_dict.append(result_summary_d_allele)

    best_tables_d = dict()
    best_results_summary = dict()

    # get the median editing activity of each score
    all_ee_results = dict()
    for result in result_summary_d_dict:
        for site_name, site_info in result.items():
            if site_name in all_ee_results.keys():
                all_ee_results[site_name].append(site_info['Editing Activity'])
            else:
                all_ee_results[site_name] = [site_info['Editing Activity']]

    # assign the "best score" into new dictionaries
    for site, ee_results in all_ee_results.items():
        median = statistics.median(ee_results)
        median_index = ee_results.index(median)

        best_tables_d[site] = tables_d_dict[median_index][site]
        best_results_summary[site] = result_summary_d_dict[median_index][site]

    return best_tables_d, best_results_summary