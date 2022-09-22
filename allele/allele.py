import numpy as np
import pandas as pd
from scipy.stats import entropy
import scipy.stats
import math
import random
from input_processing.alignment import Alignment
from utils.configurator import Configurator
import copy
from Bio import pairwise2
from Bio import Align


class AlleleForMock:
    """
    Class to handle the allelity case of the mock
    """

    def __init__(self, ratios, ref_df):
        self._ref_df = ref_df
        self._ratios = ratios  # TBD: make as a util parameter
        self._new_alleles = dict()
        self._consensus_len = None  # set the consensus of the majority length of the reads
        self._length_ratio = 0.3  # TBD: make hyperparameter. filter site with reads more than this ratio
        self._number_of_alleles = 0  # determine the number of alleles
        self._searching_bounds = 30  # TBD: make hyperparameter. The alignment will look inside the window of this number times 2 (for each side)
        self._CTC_gap = 10  # TBD: make hyperparameter. CTC - close to cut. determine the distance from gap which the alignmet method will be different
        self._half_window_len = 10  # TBD: put it in utils and import it from there. one side length of window opening: keep it even number
        # self._distance_from_cut = 10    # TBD: make hyperparameter. setting the distance from cut-position to be not counted as SNP
        self._df_mock_tx_snp_ratios = pd.DataFrame(
            columns=['site_name', 'mock_ratios', 'tx_ratios'])  # TBD change: set df for the ratios in mock vs. tx

        # set the lower decision bound for significant entropy
        self._entropy_lower_bound_score = 0
        for p in self._ratios:
            # TBD: change log2 to hyperparameter
            self._entropy_lower_bound_score -= p * math.log(p, 2)

    # -------------------------------#
    ######### Public methods #########
    # -------------------------------#

    def run(self, site_name, df):
        self._site_name = site_name
        self._df = df
        self._snp_locus = []  # holds the position of all snp loci in this site
        self._total_num_reads = 0
        self._nuc_distribution = None  # holds all distributions of all snps
        self._window = []  # holds all window that open around each snp in this site

        # TBD DELETE
        if (site_name == 'gINS11_FANCA_0029') or (site_name == 'gINS11_FANCA_115'):
            print('ok')
        # TBD: END

        df, SNP = self.mock_SNP_detection()

        # if the function finds SNPs
        if SNP:
            # insert ratios to the self variable
            curr_mock_df = pd.DataFrame(data=[[self._site_name, self._nuc_distribution, None]],
                                        columns=['site_name', 'mock_ratios', 'tx_ratios'])
            self._df_mock_tx_snp_ratios = pd.concat([self._df_mock_tx_snp_ratios, curr_mock_df], ignore_index=True)

            # iterate over all possible snp
            phases = list(self._nuc_distribution.keys())
            for i in range(self._number_of_alleles):
                allele_window_list = list()
                # filter df for reads with the current snp and clean columns
                df_for_curr_allele = df.loc[df['snp_phase'] == phases[i]]
                # get the most relevant amplicon for reference from df
                amplicon = self._get_ref_amplicon(df_for_curr_allele, site_name)
                # TBD: think what to do if it does not the same - I think i fixed it with the function above
                # indx = df_for_curr_allele.index[0]
                # if len(df_for_curr_allele.at[indx, 'alignment_w_del']) == len(
                #         self._ref_df.at[site_name, 'AmpliconReference']):
                #     amplicon = df_for_curr_allele.at[indx, 'alignment_w_del']
                # else:
                #     print('please dont')

                df_for_curr_allele = df_for_curr_allele.drop(labels=['snp_phase', 'len', 'snp_nuc_type'], axis=1)
                # prepare the window_search and the new_site_name
                for j, (window, CTC, add_before, add_after) in enumerate(self._window):
                    allele_window = window[:self._half_window_len + add_before - add_after] \
                                    + phases[i][j] + \
                                    window[add_before - add_after + self._half_window_len + 1:]
                    allele_window_list.append((allele_window, CTC, add_before, add_after))
                # TBD: change the name to more relevant one
                _new_name = self._site_name + '_' + str(self._snp_locus) + '_' + phases[i]

                # add list of:(new name, filtered df, snp positions, windows ,amplicon)
                if self._site_name in self._new_alleles.keys():
                    self._new_alleles[self._site_name].append([
                        _new_name,
                        df_for_curr_allele,
                        self._snp_locus,
                        allele_window_list,
                        amplicon])
                else:
                    self._new_alleles[self._site_name] = [[
                        _new_name,
                        df_for_curr_allele,
                        self._snp_locus,
                        allele_window_list,
                        amplicon]]

        return self._new_alleles

    # -------------------------------#
    ######### Private methods #######
    # -------------------------------#

    def _calc_entropy(self, temp_df):
        """
         Calculates entropy of the passed `pd.Series`
         :param: df: One nucleotide base through all reads in one mock site, along with frequency of each read
         :return: entropy: The entropy for this particular nucleotide base
         :return: sorted_nuc_dict: A dictionary of the nucleotides' distribution
         """
        # computing unique nuc bases
        df = temp_df.set_axis(['base', 'freq'], axis=1, inplace=False)
        nuc_bases = set(df.iloc[:, 0])
        nuc_dict = {}
        # TBD: make sure: Not eliminating '-' sign from the entropy calculation, as it can act as nuc
        # nuc_bases.discard('-')
        # computing nuc bases distribution
        for nuc in nuc_bases:
            nuc_dict[nuc] = df.loc[df['base'] == nuc, 'freq'].sum()
        sorted_nuc_dict = dict(sorted(nuc_dict.items(), key=lambda item: item[1], reverse=True))
        # take only the 2 largest items. TBD: why only the first 2??
        dict_for_entropy = {k: sorted_nuc_dict[k] for k in list(sorted_nuc_dict)[:2]}
        # calc entropy
        _entropy = scipy.stats.entropy(pd.Series(dict_for_entropy), base=2)  # get entropy from counts, log base 2

        return _entropy, sorted_nuc_dict

    def _get_ref_amplicon(self, df_for_curr_allele, site_name):
        i = 0
        indx = df_for_curr_allele.index[i]
        while len(df_for_curr_allele.at[indx, 'alignment_w_del']) != len(
                self._ref_df.at[site_name, 'AmpliconReference']):
            i += 1
            indx = df_for_curr_allele.index[i]

        if indx == len(df_for_curr_allele):
            indx = df_for_curr_allele.index[0]
            amplicon = df_for_curr_allele.at[indx, 'alignment_w_del']
        else:
            amplicon = df_for_curr_allele.at[indx, 'alignment_w_del']

        return amplicon

    def _alignment_to_return_reads(self, relevant_read, window, CTC):
        """
         Return the start and end coordinates of the alignment`
         :param: relevant_read: The relevant read to align to
         :param: window: The window around the snp
         :param: CTC: Does the snp close to cut-site?
         :return: start: Start coordinate of the alignment
         :return: end: End coordinate of the alignment
         """
        if not CTC:
            # alignment settings for not CTC (CTC = close to cut)
            local_aligner = Align.PairwiseAligner()
            local_aligner.mode = 'local'
            local_aligner.match = 5
            local_aligner.mismatch = 1
            local_aligner.open_gap_score = -100
            local_aligner.extend_gap_score = -100
            local_aligner.target_end_gap_score = 0.0
            local_aligner.query_end_gap_score = 0.0

            alignment = local_aligner.align(relevant_read, window)
        else:
            # TBD: check if the best properties and if it is the best method
            # alignment settings for  CTC (CTC = close to cut)
            global_aligner = Align.PairwiseAligner()
            global_aligner.mode = 'global'
            global_aligner.match = 5
            global_aligner.mismatch = -4
            global_aligner.open_gap_score = -10
            global_aligner.extend_gap_score = 0.0
            global_aligner.target_end_gap_score = 0.0
            global_aligner.query_end_gap_score = 0.0

            alignment = global_aligner.align(relevant_read, window)

        [read_aligned, matches, window_aligned, _] = format(alignment[0]).split("\n")
        # finding positions of start and end in the original read
        start1 = matches.find('|')
        start2 = matches.find('.')
        end1 = matches.rfind('|')
        end2 = matches.rfind('.')
        start = min(start1, start2)
        end = max(end1, end2) + 1

        return start, end

    def _get_windows(self, reference_read, cut_site):
        """
         Create list of windows to be aligned to reads with different length then consensus, and create self._window
         :param: reference_read: The reference read to extract window from
         :param: cut_site: The consensus cut-site
         :return: windows_list: List of all windows for further alignment
         """
        windows_list = list()

        for i, snp_locus in enumerate(self._snp_locus):
            additional_nb_before = 0
            additional_nb_after = 0
            CTC = False  # CTC = Close To Cut-site
            if abs(snp_locus - cut_site) <= self._CTC_gap:
                CTC = True

            # if the read is close to the end, add from the left part some more nb
            if len(reference_read) - snp_locus < self._half_window_len + 1:
                additional_nb_before = self._half_window_len - (len(reference_read) - snp_locus) + 1
                window_search = reference_read[snp_locus - self._half_window_len - additional_nb_before:snp_locus] \
                                + 'N' + reference_read[snp_locus + 1:snp_locus + 1 + self._half_window_len]
                window_for_tx_align = reference_read[snp_locus - self._half_window_len - additional_nb_before:
                                                     snp_locus + 1 + self._half_window_len]
                windows_list.append((window_search, CTC, additional_nb_before, additional_nb_after))
                self._window.append((window_for_tx_align, CTC, additional_nb_before, additional_nb_after))
            # if the read is close to the start, add from the right part some more nb
            elif snp_locus - self._half_window_len < 0:
                additional_nb_after = self._half_window_len - snp_locus
                window_search = reference_read[:snp_locus] + 'N' + \
                                reference_read[
                                snp_locus + 1:snp_locus + 1 + self._half_window_len + additional_nb_after]
                window_for_tx_align = reference_read[:snp_locus + 1 + self._half_window_len + additional_nb_after]
                windows_list.append((window_search, CTC, additional_nb_before, additional_nb_after))
                self._window.append((window_for_tx_align, CTC, additional_nb_before, additional_nb_after))
            # if "normal"
            else:
                window_search = reference_read[snp_locus - self._half_window_len:snp_locus] + 'N' + \
                                reference_read[snp_locus + 1:snp_locus + 1 + self._half_window_len]
                window_for_tx_align = reference_read[snp_locus - self._half_window_len:
                                                     snp_locus + 1 + self._half_window_len]
                windows_list.append((window_search, CTC, additional_nb_before, additional_nb_after))
                self._window.append((window_for_tx_align, CTC, additional_nb_before, additional_nb_after))
        return windows_list

    def _return_reads_to_nuc_dist(self, df, filtered_df, windows_list):
        """
         Return reads with different length than the consensus to the df
         :param: df: The df with the reads with the same length
         :param: filtered_df: The df of the reads that were filtered due to different length of the consensus
         :param: windows_list: list of all windows to be aligned to each read
         :return: df: The df with some returned reads (that were able to be returned)
         :return: df_dropped: Reads that were not able to be returned
         """
        df_dropped = pd.DataFrame(columns=list(filtered_df.columns))
        num_reads_filtered_out = 0

        for i, row in filtered_df.iterrows():
            read = row['alignment_w_del']
            for idx, (window, CTC, additional_nb_before, additional_nb_after) in enumerate(windows_list):
                # prepare the relevant sequence to align to
                if self._snp_locus[idx] - self._searching_bounds < 0:
                    relevant_read = read[:self._searching_bounds * 2]
                elif self._snp_locus[idx] + self._searching_bounds > len(read):
                    relevant_read = read[-self._searching_bounds * 2:]
                else:
                    relevant_read = read[self._snp_locus[idx] - self._searching_bounds:
                                         self._snp_locus[idx] + self._searching_bounds]
                # get the start and end coordinates according to the alignment
                start, end = self._alignment_to_return_reads(relevant_read, window, CTC)
                # if succeed to align properly # TBD: Check the case of CTC==True
                if end - start == len(window):  # TBD: maybe to insert this part into self._alignment_to_return_reads()
                    snp_nb = relevant_read[start:end][
                        self._half_window_len + additional_nb_before - additional_nb_after]
                    filtered_df.at[i, 'snp_nuc_type'].append(snp_nb)
                else:
                    num_reads_filtered_out += row['frequency']
                    # TBD: add to log
                    # only rows that all snps have been found # TBD: check with Zohar if ok
                    # TBD: ask zohar: what happen if the SNP is at the end and thus read is shorter
                    df_dropped = pd.concat([df_dropped, pd.DataFrame(row).transpose()])
                    filtered_df = filtered_df.drop(i)
                    break

        df = pd.concat([df, filtered_df])

        return df, df_dropped

    def _compute_lengths_distribution(self, sorted_df, possible_len):
        """
         Compute the distribution of the length of all the *aligned* reads in the df
         :param sorted_df: sorted df by frequency
         :param possible_len: list of all possible lengths in the df
         :return: lengths_ratio: The ratio of the largest length percentage
         :return: len_dict: Dictionary of lengths percentages
         """
        len_dict = {}
        for i_len in possible_len:
            len_dict[i_len] = sorted_df.loc[sorted_df['len'] == i_len, 'frequency'].sum()
        sum_len = sum(len_dict.values())
        values_len = list(len_dict.values())
        percent_len = [x / sum_len for x in values_len]
        lengths_ratio = np.sort(percent_len)[::-1]

        return lengths_ratio, len_dict

    def _find_all_potential_snp(self, df):
        """
         Find all potential snp among the filtered df
         :param df: The df with the additional information regarding snps, filtered with the same read length
         :return: df: return the same df as self._df however with additional columns regarding the snps
         :return: SNP: True/False to indicate if SNP is exists or not
         """
        # obtain entropies for the site run
        local_score = []
        freq_list = df['frequency']

        # split the DataFrame into separated columns to calculate snps
        parsed_df = df['alignment_w_del'].apply(lambda x: pd.Series(list(x)))  # TBD: expensive in time

        # iterate over all columns, calc entropy and append to local score
        for i in range(len(parsed_df.columns)):
            temp_df = pd.concat([parsed_df[i], freq_list], axis=1, join='inner')
            temp_score, _ = self._calc_entropy(temp_df)
            local_score.append(temp_score)

            # if score is higher >> it is a snp
            if temp_score >= self._entropy_lower_bound_score:
                self._snp_locus.append(i)
            else:
                continue

        # get all snps to column 'snp_nuc_type'
        for i_row, row in df.iterrows():
            for snp_loc in self._snp_locus:
                df.at[i_row, 'snp_nuc_type'].append(str(row['alignment_w_del'][snp_loc]))

        return df

    def _norm_nuc_distribution(self, df):
        """
         Find all snps and return df with the snp bases
         :param: df: The site df with the returned reads
         :return: "self" assignments
         """
        # create dictionary and assign values inside
        nuc_dict = dict()
        for i, row in df.iterrows():
            phase = df.at[i, 'snp_phase']
            freq = df.at[i, 'frequency']
            if phase in nuc_dict.keys():
                nuc_dict[phase] += freq
            else:
                nuc_dict[phase] = freq

        # normalized the dictionary of the nucleotide distribution
        self._total_num_reads = sum(nuc_dict.values())
        for snp in nuc_dict.keys():
            nuc_dict[snp] /= self._total_num_reads
        self._nuc_distribution = nuc_dict

    def _get_most_frq_cut_site(self, same_len_df):
        cut_site_dict = dict()
        freq = list(same_len_df['frequency'])
        cut_sites = list(same_len_df['alignment_cut_site'])
        for i, cut in enumerate(cut_sites):
            if cut in cut_site_dict.keys():
                cut_site_dict[cut] += freq[i]
            else:
                cut_site_dict[cut] = freq[i]

        return int(max(cut_site_dict, key=cut_site_dict.get))

    def _get_num_alleles(self):
        """
         Calculate the optimal number of alleles in a given distribution
         :param: self: Get some self variables
         :return: coverage_list: The list of coverage
         :return: entropy_scores: The scores of all entropies of all number of potential alleles
         :return: norm_scores_list: Score after normalization
         :return: number_of_alleles: The optimum number of alleles
         """
        entropy_scores = list()
        norm_scores_list = list()
        coverage_list = list(self._nuc_distribution.values())

        for i in range(1, len(coverage_list)):
            curr_vals = coverage_list[:i + 1]
            _entropy = scipy.stats.entropy(pd.Series(curr_vals), base=i + 1)
            entropy_scores.append(_entropy)
            norm_score = self._score_normalization(_entropy, sum(curr_vals))
            norm_scores_list.append(norm_score)

        number_of_alleles = np.argmax(norm_scores_list) + 2

        return coverage_list, entropy_scores, norm_scores_list, number_of_alleles

    def _score_normalization(self, _entropy, _coverage):
        # TBD: optimize the ratios between coverage and entropy
        """
         Find all snps and return df with the snp bases
         :param: _entropy: The entropy of the current number of alleles
         :param: _coverage: The current coverage of these number of alleles out of 100%
         :return: Normalized calculation between coverage and entropy
         """
        # TBD: make hyperparameter
        return math.pow(_coverage, 1 / 10) * math.pow(_entropy, 9 / 10)

    # def _get_window_for_multi_snp(self, ref_read_for_window, cut_site):
    #     """
    #      Get all windows for further alignment for multi-snp
    #      :param: ref_read_for_window: Reference read to get window from
    #      :param: cut_site: The cut site
    #      :return: windows_list: List of all windows - one per snp
    #      """
    #     snps = copy.deepcopy(self._snp_locus)
    #     # TBD: ask zohar
    #     # Excluding the SNP close to the cut-site
    #     # window_for_elimination = [cut_site - self._distance_from_cut,
    #     #                           cut_site + self._distance_from_cut]
    #     # for i, snp_multi_loc in enumerate(snps):
    #     #     if (snp_multi_loc >= window_for_elimination[0]) and (snp_multi_loc <= window_for_elimination[1]):
    #     #         snps.pop(i)
    #     # TBD add to log the ones that out
    #
    #     # create windows
    #     windows_list = list()
    #     for snp in snps:
    #         temp_window = ref_read_for_window[snp - self._half_window_len:
    #                                           snp + self._half_window_len + 1]
    #         windows_list.append(temp_window)
    #
    #     # TBD: Zohar said not necessary!!!
    #     # merging close SNPs to one phase
    #     # i, j = 0, 0
    #     # while j < len(snps):
    #     #     j += 1
    #     #     if j+1 > len(snps):
    #     #         temp_window = ref_read_for_window[snps[j - 1] - self._half_window_len:
    #     #                                             snps[j - 1] + self._half_window_len + 1]
    #     #         windows_list.append(temp_window)
    #     #     elif snps[i] + self._half_window_len > snps[j]:
    #     #         continue
    #     #     else:
    #     #         i += 1
    #     #         if i == j:
    #     #             temp_window = ref_read_for_window[snps[i-1]-self._half_window_len :
    #     #                                              snps[i-1]+self._half_window_len+1]
    #     #             windows_list.append(temp_window)
    #     #         else:
    #     #             temp_window = ref_read_for_window[snps[i - 1] - self._half_window_len:
    #     #                                                 snps[j] + self._half_window_len + 1]
    #     #             windows_list.append(temp_window)
    #
    #     return windows_list

    def mock_SNP_detection(self):
        """
         Find all snps and return df with the snp bases
         :param: self: Get some self variables
         :return: df: return the same df as self._df however with additional columns regarding the snps
         :return: SNP: True/False to indicate if SNP is exists or not
         """
        SNP = False
        false_df = False
        try:
            df = self._df
            df.loc[:, 'snp_nuc_type'] = [list() for x in range(len(df.index))]  # create snp list for each row
            df = df.sort_values(['frequency'], ascending=False)  # sort according to frequency
            original_instance_number = sum(df['frequency'])  # count the number of original reads

            # adding length of each read to each row
            len_reads = list(df['alignment_w_del'].str.len())
            df['len'] = len_reads

            # compute the distribution of the length of the reads
            lengths_ratio, len_dict = self._compute_lengths_distribution(df, set(len_reads))

            # TBD confirm: if the length of the reads differ and have sort-of uniform distribution
            # if the highest freq of length is smaller than 70% than filter out all 30%
            ambiguous_reads_length = False
            if lengths_ratio[0] < (1 - self._length_ratio):
                ambiguous_reads_length = True
                # TBD ADD: add to log - note that filter a {percentage of reads} from this site
                # TBD ADD: create a file of ambiguous reads in the site directory

            # calc the majority number of reads by length and filter Dataframe based on it
            self._consensus_len = max(len_dict, key=len_dict.get)
            mask = df['len'] == self._consensus_len
            same_len_df = df.loc[mask]
            filtered_reads_diff_len = df.loc[~mask]

            # find all potential stand-alone snps
            same_len_df = self._find_all_potential_snp(same_len_df)  # TBD: takes long time - fix
            cut_site = self._get_most_frq_cut_site(same_len_df)

            # TBD: make 5 hyperparameter. but can't be too big because it cannot create directory with too long name
            if (len(self._snp_locus) > 0) and (len(self._snp_locus) < 5):
                SNP = True
                sum_read_before_adding = sum(same_len_df['frequency'])
                # handle reads that are not at length of the self._consensus length
                reference_read = same_len_df.loc[list(same_len_df['frequency']).index(max(same_len_df['frequency'])),
                                                 'alignment_w_del']
                windows_list_for_realignment = self._get_windows(reference_read, cut_site)
                df_w_returned, df_dropped = self._return_reads_to_nuc_dist(
                                            same_len_df, filtered_reads_diff_len, windows_list_for_realignment)

                # TBD ADD: df_dropped to the site dictionary as an output
                sum_read_after_adding = sum(df_w_returned['frequency'])
                # TBD: add to log - {sum_read_after_adding-sum_read_before_adding} added during length processing

                # phasing the multi-snp into one phase instead of a list of snps
                df_w_returned['snp_phase'] = df_w_returned['snp_nuc_type'].apply(lambda x: ''.join(x))
                # normalized the snp distribution # TBD: a little time-consuming. check
                self._norm_nuc_distribution(df_w_returned[['snp_phase', 'frequency']])
                # get the number of alleles
                _, _, _, self._number_of_alleles = self._get_num_alleles()

                return df_w_returned, SNP

                # # open a nucleotide window for future searching in tx reads
                # ref_read_for_window = same_len_df['alignment_w_del'][0]
                # # if there is only 1 snp in the df
                # if len(self._snp_locus) == 1:
                #     # TBD: Confirm with Zohar, for now - blocked
                #     # window_for_elimination = [cut_site - self._distance_from_cut,
                #     #                           cut_site + self._distance_from_cut]
                #     # if (self._snp_locus[0] >= window_for_elimination[0]) and
                #     #    (self._snp_locus[0] <= window_for_elimination[1]):
                #     #     # TBD add to log the site
                #     #     print('cannot be a site due to CUT')
                #     #     return false_df, SNP
                #     self._window = [ref_read_for_window[self._snp_locus[0] - self._half_window_len:
                #                                         self._snp_locus[0] + self._half_window_len + 1]]
                # # if multi-snp
                # elif len(self._snp_locus) > 1:
                #     self._window = self._get_window_for_multi_snp(ref_read_for_window, cut_site)
                # else:
                #     return false_df, SNP

            else:
                return false_df, SNP
        except:
            return false_df, SNP


class AlleleForTx:
    """
    Class to handle the allele case of the treatment
    """

    def __init__(self, tx_df, new_alleles):
        self._tx_df = tx_df
        self._new_alleles = new_alleles
        # create list of sites to be handled
        self._sites_to_be_alleles = list(self._new_alleles.keys())
        self._new_tx_df = dict()
        self._sites_score = pd.DataFrame(columns=['site_name', 'avg_score'])
        self.uncertain_reads = dict()

    # -------------------------------#
    ######### Public methods #########
    # -------------------------------#

    def run(self, ratios_df):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        for tx_site_name, tx_site_df in self._tx_df.items():
            print(tx_site_name)
            if tx_site_name == 'gINS11_FANCA_154':
                print('ok')
            if tx_site_name in self._sites_to_be_alleles:  # if tx site an allele site
                self._new_tx_df[tx_site_name] = []
                tx_site_df['allele'] = None
                tx_site_df['alignment_score'] = math.nan
                uncertain_reads = pd.DataFrame(columns=tx_site_df.columns)
                list_sites_of_allele = self._new_alleles[tx_site_name]  # list of all relevant allele dfs for this site
                windows = list()
                new_sites_name = list()
                for site_allele in list_sites_of_allele:  # iterate over all possible alleles
                    windows.append(site_allele[3])  # add the window to list
                    new_sites_name.append(site_allele[0])  # add the new name to list
                for i, row in tx_site_df.iterrows():  # iterate over all rows in df and assign to relevant allele
                    scores = dict()
                    seq = row['alignment_w_del']
                    for allele_windows in windows:
                        for j, SNP_window in enumerate(allele_windows):
                            _, curr_score = get_snp_locus_and_score(seq, SNP_window)
                            # add score of alignment to a list of the same window
                            if j in scores.keys():
                                scores[j].append(curr_score)
                            else:
                                scores[j] = [curr_score]
                    best_idx, index_list = self._determine_best_idx(scores)  # find the best idx or return False

                    if type(best_idx) == int:  # If all SNP positions agrees on one window phase
                        tx_site_df.loc[i, 'allele'] = new_sites_name[best_idx]
                        alignment_score = self._get_alignment_score(best_idx, scores)
                        tx_site_df.loc[i, 'alignment_score'] = alignment_score
                    else:  # if there is a disagreement upon the phase
                        # reshaping row and appending
                        row_to_append = pd.DataFrame(row.values.reshape(1, len(tx_site_df.columns))[0]).T
                        row_to_append.rename(index={0: i}, inplace=True)
                        row_to_append.columns = tx_site_df.columns
                        row_to_append.loc[
                            i, 'alignment_score'] = index_list  # raise an numpy.VisibleDeprecationWarning error. canceled it the error
                        uncertain_reads = pd.concat([uncertain_reads, row_to_append])

                ratios_dict = dict()
                # set the new df for the tx allele
                for new_site in new_sites_name:
                    # get the tx ratios for this site
                    # TBD - change to more convenient method to retrieve SNP type
                    site_snp = new_site[new_site.rfind('_') + 1:]
                    tx_site_df_temp = tx_site_df.loc[tx_site_df['allele'] == new_site]
                    sum_reads = sum(list(tx_site_df_temp['frequency']))
                    ratios_dict[site_snp] = sum_reads
                    # calculate the alignment average score. If less than one [TBD]
                    avg_score = tx_site_df_temp['alignment_score'].mean(axis=0, skipna=True)
                    self._sites_score = self._sites_score.append({'site_name': new_site, 'avg_score': avg_score},
                                                                 ignore_index=True)
                    tx_site_df_temp = tx_site_df_temp.drop(labels='allele', axis=1)
                    tx_site_df_temp = tx_site_df_temp.drop(labels='alignment_score', axis=1)
                    self._new_tx_df[tx_site_name].append([new_site, tx_site_df_temp])
                # TBD: insert to log:
                self.uncertain_reads[tx_site_name] = [sum(uncertain_reads['frequency']),
                                                      sum(tx_site_df['frequency'])]

                # return uncertain_reads back [TBD] - should it be after or before the tx_ratios?
                # bring_back = 2
                # if bring_back == 1:
                #     # first method - by the ratio of the mock
                #     self.merge_back_uncertain_read1(tx_site_name, uncertain_reads, ratios_df)
                # if bring_back == 2:
                #     # second method - return by aligning to the consensus in each site
                #     self.merge_back_uncertain_read2(tx_site_name, uncertain_reads)

                # normalize the dict
                try:
                    sum_all_reads = sum(ratios_dict.values())
                    for _i in ratios_dict.keys():
                        ratios_dict[_i] /= sum_all_reads
                except:
                    print(f'problem with {tx_site_name}')

                # append to df
                ratios_df.at[list(ratios_df['site_name']).index(tx_site_name), 'tx_ratios'] = ratios_dict

        return self._new_tx_df, round(self._sites_score, 3), ratios_df

    # -------------------------------#
    ######### Private methods #######
    # -------------------------------#

    def merge_back_uncertain_read1(self, tx_site_name, uncertain_reads, ratios_df):
        random.seed(10)
        temp_dict = dict(ratios_df.loc[ratios_df['site_name'] == tx_site_name]['mock_ratios'])
        site_mock_dist = temp_dict[list(temp_dict.keys())[0]]
        # [TBD: add the number of alleles to extract the correct number)
        alleles = 2
        SNP_type = []
        SNP_dist = []
        for i in range(alleles):
            SNP_type.append(list(site_mock_dist.keys())[i])
            SNP_dist.append(list(site_mock_dist.values())[i])
        # norm SNP_dist
        sum_SNP_dist = sum(SNP_dist)
        for dist in SNP_dist:
            dist /= sum_SNP_dist

    def merge_back_uncertain_read2(self, tx_site_name, uncertain_reads):
        consensus_read = []
        site_names = []
        new_sites_dfs = self._new_tx_df[tx_site_name]
        for param in new_sites_dfs:
            site_names.append(param[0])
            df = param[1]
            consensus_read.append(list(df[df['frequency'] == max(list(df['frequency']))]['alignment_w_del'])[0])
        for i, row in uncertain_reads.iterrows():
            read = row['alignment_w_del']
            scores = []
            for idx, reference in enumerate(consensus_read):
                _, curr_score = get_snp_locus_and_score(reference, read)
                scores.append(curr_score)

            uncertain_reads.loc[i, 'allele'] = site_names[np.argmin(scores)]
            uncertain_reads.loc[i, 'alignment_score'] = np.min(scores)

        return uncertain_reads

    def _determine_best_idx(self, scores):
        index_list = list()
        for score in scores.values():
            min_score = min(score)
            all_min_idx = [idx for idx, val in enumerate(score) if val == min_score]
            index_list.append(all_min_idx)
        best_indexes = list(set(index_list[0]).intersection(*index_list))

        if len(best_indexes) == 1:
            return best_indexes[0], index_list
        else:
            return False, index_list

    def _get_alignment_score(self, indx, scores):
        sum_all = 0
        for lst in scores.values():
            sum_all += lst[indx]
        return sum_all / len(scores)


class ref_dfAlleleHandler:
    """
    Class to handle the creation of new ref_df with allele sites
    """

    def __init__(self):
        self._ref_df = None

    # -------------------------------#
    ######### Public methods #########
    # -------------------------------#

    def run(self, ref_df, new_allele_sites):
        self._ref_df = ref_df
        for original_site_name, sites_list in new_allele_sites.items():
            for site_allele in sites_list:
                new_site_name = site_allele[0]
                SNP_window = site_allele[3]
                loci = site_allele[2]
                new_amplicon = site_allele[4]
                # new_amplicon = self.create_new_amplicon(new_site_name,original_site_name,SNP_window,loci)
                new_row_for_ref_df = self._ref_df[self._ref_df['Site Name'] == original_site_name]
                new_row_for_ref_df = new_row_for_ref_df.rename(index={original_site_name: new_site_name})
                new_row_for_ref_df.loc[new_site_name, 'AmpliconReference'] = new_amplicon
                new_row_for_ref_df.loc[new_site_name, 'Site Name'] = new_site_name
                self._ref_df = pd.concat([self._ref_df, new_row_for_ref_df])
        '''add PAM coordinates and grna coordinates to df'''
        self.get_PAM_site()

        return self._ref_df

    # -------------------------------#
    ######### Private methods #######
    # -------------------------------#

    def create_new_amplicon(self, new_site_name, original_site_name, SNP_window, loci):
        curr_amplicon = str(self._ref_df[self._ref_df['Site Name'] == original_site_name]['AmpliconReference'][0])
        SNP_locus, SNP_score = get_snp_locus_and_score(curr_amplicon, SNP_window)
        new_amplicon = curr_amplicon[:SNP_locus] + new_site_name[-1] + curr_amplicon[SNP_locus + 1:]
        '''TBD: print to be deleted'''
        print(f'the gap between SNP and new SNP locus is: {loci} - {SNP_locus} = {loci - SNP_locus}')

        return new_amplicon

    def get_PAM_site(self):
        self._ref_df['PAM_window'] = None
        self._ref_df['grna_window'] = None

        for i, row in self._ref_df.iterrows():
            cut_site = row['cut-site']
            if row['sgRNA_reversed'] == True:
                PAM = [cut_site - 6, cut_site - 4]
                grna = [cut_site - 3, cut_site + 16]
            elif row['sgRNA_reversed'] == False:
                PAM = [cut_site + 3, cut_site + 5]
                grna = [cut_site - 17, cut_site + 2]

            self._ref_df.at[i, 'PAM_window'] = PAM
            self._ref_df.at[i, 'grna_window'] = grna


# -------------------------------#
######## Global Functions ########
# -------------------------------#
def count_diff(seq_A, seq_B):
    counter = 0
    for i, j in zip(seq_A, seq_B):
        if i != j:
            counter += 1
    return counter


def get_snp_locus_and_score(curr_amplicon, SNP_window):
    smallest_gap = np.inf
    i_smallest = None
    for i in range(len(curr_amplicon) - (len(SNP_window) - 1)):
        try:
            window = curr_amplicon[i: i + len(SNP_window)]
        except:
            break
        curr_gap_score = count_diff(SNP_window, window)
        if curr_gap_score < smallest_gap:
            smallest_gap = curr_gap_score
            i_smallest = i
    # TBD CHANE: change 10 to self._half_window_len which will be in utils
    return i_smallest + 10, smallest_gap


def align_allele_df(reads_df, ref_df, amplicon_min_score, translocation_amplicon_min_score):
    # set configuration and aligner for the alignment
    _cfg = Configurator.get_cfg()
    _aligner = Alignment(_cfg["alignment"], amplicon_min_score, translocation_amplicon_min_score,
                         _cfg["NHEJ_inference"]["window_size"])

    new_sites = reads_df.copy()

    # iterate over each new allele and re-align it
    for site, site_allele in reads_df.items():
        for i in range(len(site_allele)):
            allele_name = site_allele[i][0]
            allele_df = site_allele[i][1]
            # zero all aligned so far

            cut_site = ref_df.loc[allele_name, 'cut-site']
            ref_amplicon = ref_df.loc[allele_name, 'AmpliconReference']

            new_allele_df = _aligner.align_reads(allele_df, ref_amplicon, cut_site, None, None, None, None, allele=True)
            new_sites[site][i][1] = new_allele_df

    return new_sites
