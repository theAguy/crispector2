import pandas as pd
import numpy as np
import copy
import os
from scipy.spatial import distance
import matplotlib.pyplot as plt
from typing import Dict
from algorithm.binomial_probability import compute_binom_p
from crispector2.algorithm.core_algorithm import CoreAlgorithm
from utils.constants_and_types import FREQ, TX_READ_NUM, MOCK_READ_NUM, ON_TARGET, CUT_SITE, AlgResult, IS_RANDOM, \
    EDIT_PERCENT, RANDOM_EDIT_READS, TX_EDIT, PAM_WINDOW, GRNA_WINDOW
from modifications.modification_tables import ModificationTables
from modifications.modification_types import ModificationTypes
from scipy.stats import norm


# functions and methods for determine sites to be re-calculated


class Interval:
    """
    class to handle intervals of CI of results summary
    """

    def __init__(self, interval):
        self.start = interval[0]
        self.end = interval[1]


def _is_intersect(ci_arr):
    """
    function that returns True if some intervals are intersects and False if all not intersects.
    :param: ci_arr: array of intervals
    :return: bool: True / False intersection of intervals
    """
    n = len(ci_arr)
    ci_arr.sort(key=lambda x: x.start)
    # In the sorted array, if the start of an interval is less than end of previous one - there is an overlap
    for i in range(1, n):
        if ci_arr[i - 1].end > ci_arr[i].start:
            return True
    # Else - no overlap
    return False


def _compute_confidence_interval(editing_activity, n_reads_tx, confidence_interval):
    """
    Compute confidence interval and returns low & high CI boundary
    :param editing_activity: the calculated editing activity
    :param n_reads_tx: number of treatment reads
    :param confidence_interval: the confidence interval parameter
    :return: Tuple of low & high CI boundary
    """
    confidence_inv = norm.ppf(confidence_interval + (1 - confidence_interval) / 2)
    half_len_CI = confidence_inv * np.sqrt((editing_activity * (1 - editing_activity)) / n_reads_tx)
    return max(0, editing_activity - half_len_CI), editing_activity + half_len_CI


def _compute_new_statistics_based_on_random_assignment(original_n_reads_tx, original_n_reads_edited, allele_random,
                                                       sum_all, confidence_interval):
    """
    compute new high and low CI, where for high we are taking into consideration all random reads, and for low none
    :param original_n_reads_tx: the original number of treatment reads
    :param original_n_reads_edited: the original number of edited treatment reads
    :param allele_random: number of random edited reads of the specific site
    :param sum_all: sum of all random edited reads from all alleles of the same site
    :param confidence_interval: the confidence interval parameter
    :return: array of [low_CI, high_CI]
    """
    # high_confidence - consider all random edited reads from both alleles
    n_reads_tx_high_CI = original_n_reads_tx + sum_all - allele_random
    n_reads_edited_high_CI = original_n_reads_edited + sum_all - allele_random
    editing_activity_high_CI = (n_reads_edited_high_CI / n_reads_tx_high_CI)
    _, CI_high = _compute_confidence_interval(editing_activity_high_CI, n_reads_tx_high_CI, confidence_interval)
    # low_confidence - subtract random edited reads
    n_reads_tx_low_CI = original_n_reads_tx - allele_random
    n_reads_edited_low_CI = original_n_reads_edited - allele_random
    editing_activity_low_CI = n_reads_edited_low_CI / n_reads_tx_low_CI
    CI_low, _ = _compute_confidence_interval(editing_activity_low_CI, n_reads_tx_low_CI, confidence_interval)

    return [CI_low * 100, CI_high * 100]


def estimate_random_reads_editing_effect(dict_of_alleles, confidence_interval):
    """
    function that checks if site has alleles that CI-overlap after manipulation of:
        high_CI = all random edited reads assign to an allele
        low_CI = none random edited reads assign to an allele
    :param dict_of_alleles: dictionary of all alleles
    :param confidence_interval: the confidence interval parameter
    :return: True if the alleles of a site are overlapping, and False if not
    """
    new_alleles_CI = list()
    try:
        sum_randoms = sum(allele[RANDOM_EDIT_READS] for allele in dict_of_alleles.values())
    except:
        # read count per allele is too low that there is no statistics for it
        return False

    for allele in dict_of_alleles.values():
        n_allele_random = allele[RANDOM_EDIT_READS]
        n_reads_tx = allele[TX_READ_NUM]
        n_reads_edited = allele[TX_EDIT]
        new_allele = _compute_new_statistics_based_on_random_assignment(n_reads_tx, n_reads_edited,
                                                                        n_allele_random, sum_randoms,
                                                                        confidence_interval)
        new_alleles_CI.append(Interval(new_allele))
    if _is_intersect(new_alleles_CI):
        return True
    return False


#############################################################################################################
#############################################################################################################

# functions for re-calculating sites' editing activity


def _separate_random_reads_from_dfs(tables_data, re_run_sites):
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

            # tx
            try:
                # take the df
                tx_df = tables_data[allele_name].tx_reads
                # filter the random reads
                random_tx_df = tx_df[tx_df[IS_RANDOM] == True]
                # assign to the original site. combine with other alleles random reads
                if site in random_tx.keys():
                    concat_df = pd.concat([random_tx[site], random_tx_df])
                    random_tx[site] = concat_df
                else:
                    random_tx[site] = random_tx_df
                # remove from the current df
                random_index = tx_df[tx_df[IS_RANDOM] == True].index
                tables_data[allele_name].tx_reads.drop(random_index, inplace=True)

            # if there are no random reads - pass this stage
            except:
                pass

            # mock
            try:
                # take the df
                mock_df = tables_data[allele_name].mock_reads
                # filter the random reads
                random_mock_df = mock_df[mock_df[IS_RANDOM] == True]
                # assign to the original site. combine with other alleles random reads
                if site in random_mock.keys():
                    concat_df = pd.concat([random_mock[site], random_mock_df])
                    random_mock[site] = concat_df
                else:
                    random_mock[site] = random_mock_df
                # remove from the current df
                random_index = mock_df[mock_df[IS_RANDOM] == True].index
                tables_data[allele_name].mock_reads.drop(random_index, inplace=True)

            # if there are no random reads - pass this stage
            except:
                pass

    return tables_data, random_mock, random_tx, site_to_alleles_link


def _randomly_assign_random_reads_to_dfs(tables_d, random_tx_dict, random_mock_dict, link_site_allele):
    """
    Randomly assign "random reads" to relevant alleles
    :param tables_d: original tables, without "random reads" in their dfs
    :param random_tx_dict: per each site, all the tx "random reads"
    :param random_mock_dict: per each site, all the mock "random reads"
    :param link_site_allele: the mapping between site to alleles
    :return: new tables_d with new random assignment of "random reads"
    """

    new_tables_d = copy.deepcopy(tables_d)

    # assign randomly to mock
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

    # assign randomly to tx
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
                        translocation_amplicon_min_score, binom_p_d):
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
    :param binom_p_d: the same binom_p_d as the first full run
    :return: The new dfs of the alleles and the new statistics scores
    """

    tx_reads_d = dict()
    mock_reads_d = dict()
    allele_sites = list()

    # create tx_reads_d and mock_reads_d
    for site, alleles in re_run_overlapping_sites.items():
        for allele_name, allele_info in alleles.items():
            allele_sites.append(allele_info[0])

        for allele in allele_sites:
            tx_reads_d[allele] = dfs_data[allele].tx_reads.reset_index()
            mock_reads_d[allele] = dfs_data[allele].mock_reads.reset_index()

    # No need for re-align as the CIGAR would be the same.
    # if not, thus this read is not random because it has better alignment

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
            result_summary_d_allele[allele][ON_TARGET] = row[ON_TARGET]
            result_summary_d_allele[allele][PAM_WINDOW] = allele_ref_df.at[site, PAM_WINDOW]
            result_summary_d_allele[allele][GRNA_WINDOW] = allele_ref_df.at[site, GRNA_WINDOW]

    return tables_d_allele, result_summary_d_allele


# def re_align_df(reads_df, ref_df, amplicon_min_score, translocation_amplicon_min_score):
#     """
#     Re-align df to the relevant amplicon
#     :param reads_df:
#     :param ref_df:
#     :param amplicon_min_score: for re-alignment
#     :param translocation_amplicon_min_score: for re-alignment
#     :return: Re-aligned df
#     """
#     # set configuration and aligner for the alignment
#     _cfg = Configurator.get_cfg()
#     _aligner = Alignment(_cfg["alignment"], amplicon_min_score, translocation_amplicon_min_score,
#                          _cfg["NHEJ_inference"]["window_size"])
#     new_sites = reads_df.copy()
#
#     # iterate over each new allele and re-align it
#     for site, df in reads_df.items():
#         cut_site = ref_df.loc[site, 'cut-site']
#         ref_amplicon = ref_df.loc[site, 'AmpliconReference']
#
#         new_allele_df = _aligner.align_reads(df, ref_amplicon, cut_site,
#                                              None, None, site, None, allele=True)
#         new_sites[site] = new_allele_df
#
#     return new_sites


def _get_medoid(d, sites_linkage, outdir):
    """
    function that takes all iterations of summary results and select the medoid of the data with respect to
    each site's alleles
    :param d: the dictionary with all statistics results of all iterations
    :param sites_linkage: a key-value of all sites and their alleles
    :param outdir: path of out directory
    :return: The medoid of the results. medoid per site (same alleles in the site would get same medoid)
    """

    best_index_per_allele = dict()

    for site, alleles in sites_linkage.items():
        # get all results as data points in the dimension of number of alleles
        data_points = list()
        for indx, results in enumerate(d):
            sub_point = list()
            for allele_name in alleles:
                try:
                    editing_activity = results[allele_name][EDIT_PERCENT]
                except:  # for read count less than threshold
                    editing_activity = 0
                sub_point.append(editing_activity)
            data_points.append(sub_point)

        # creating an empty distances matrix
        dict_len = len(d)
        dist_matrix = np.empty(shape=(dict_len, dict_len), dtype='object')

        # calculate each pairwise euclidean distance
        for i, fix_point in enumerate(data_points):
            for j, var_point in enumerate(data_points):
                dist_matrix[i, j] = distance.euclidean(fix_point, var_point)

        # get the smallest aggregated distance as the medoid and as the selected point.
        medoid = np.argmin(dist_matrix.sum(axis=0))

        for allele_name in alleles:
            best_index_per_allele[allele_name] = medoid

        # plotting the data points
        if len(alleles) == 2:
            x = [row[0] for row in data_points]
            y = [row[1] for row in data_points]
            label = [idx for idx in range(len(data_points))]
            plt.scatter(x, y)

            for i, txt in enumerate(label):
                plt.annotate(txt, (x[i], y[i]))
            plt.title(f'The medoid of site {site} is: {medoid}')
            plt.savefig(os.path.join(outdir, f'medoid_{site}.png'), box_inches='tight')
            plt.close()
            plt.clf()

    return best_index_per_allele


def re_calculate_statistics(outdir, enable_substitutions, confidence_interval, donor, min_num_of_reads,
                            override_noise_estimation, allele_ref_df, dfs_data, re_run_overlapping_sites,
                            amplicon_min_score, translocation_amplicon_min_score, binom_p_d):
    """
    function that takes all ambiguous sites (regarding CI), re-assign the random reads to the alleles and re-compute
    the statistics. This done 11 times, taking the median score of each site as the "best" score
    :param outdir: path of out directory
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
    :param binom_p_d: the same binom_p_d as the first full run
    :return: The best selected statistics and dfs out of 11 random possibles
    """
    # take out from mock and treatment dfs all the "random reads" and store them separately
    tables_d_wo_random, random_mock_dict, random_tx_dict, map_site_allele = \
        _separate_random_reads_from_dfs(dfs_data, re_run_overlapping_sites)

    tables_d_dict = list()
    result_summary_d_dict = list()
    for i in range(11):
        # assign "random reads" randomly between sites
        tables_d_w_random = _randomly_assign_random_reads_to_dfs(tables_d_wo_random, random_tx_dict, random_mock_dict,
                                                                 map_site_allele)

        # compute new statistics per each site that was modified randomly
        tables_d_allele, result_summary_d_allele = compute_best_stats(enable_substitutions, confidence_interval, donor,
                                                                      min_num_of_reads, override_noise_estimation,
                                                                      allele_ref_df, tables_d_w_random,
                                                                      re_run_overlapping_sites,
                                                                      amplicon_min_score,
                                                                      translocation_amplicon_min_score, binom_p_d)
        # store the dfs and the statistics results
        tables_d_dict.append(tables_d_allele)
        result_summary_d_dict.append(result_summary_d_allele)

    # calculate the medoid of each site to set as the best result
    best_index_per_allele = _get_medoid(result_summary_d_dict, map_site_allele, outdir)

    # assign the "best score" into new dictionaries
    best_tables_d = dict()
    best_results_summary = dict()
    for allele, medoid_index in best_index_per_allele.items():
        best_tables_d[allele] = tables_d_dict[medoid_index][allele]
        best_results_summary[allele] = result_summary_d_dict[medoid_index][allele]

    return best_tables_d, best_results_summary
