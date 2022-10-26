import numpy as np
import pandas as pd
from scipy.stats import entropy
import scipy.stats
import math
import random
from input_processing.alignment import Alignment
from utils.configurator import Configurator
# import copy
# from Bio import pairwise2
from Bio import Align


class AlleleForMock:
    """
    Class to handle the allele case of the Mock
    """

    def __init__(self, ratios, ref_df):
        self.reads_drop_down = list()
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
        self.df_mock_tx_snp_ratios = pd.DataFrame(
            columns=['site_name', 'mock_ratios_before', 'mock_ratios_after',
                     'tx_ratios', 'number_of_alleles'])  # TBD change: set df for the ratios in mock vs. tx

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
        self._nuc_distribution_before = None  # holds all distributions of all snps
        self._nuc_distribution_after = None
        self._window = []  # holds all window that open around each snp in this site

        # TBD DELETE
        if (site_name == 'gINS11_FANCA_129'):
            print('ok')
        # TBD: END

        df, SNP = self._mock_SNP_detection()

        # if the function finds SNPs
        if SNP:
            # insert ratios to the self variable. TBD: move to the end and normalized by number of alleles
            curr_mock_df = pd.DataFrame(data=[[self._site_name, self._nuc_distribution_before,
                                               self._nuc_distribution_after, None, self._number_of_alleles]],
                                        columns=['site_name', 'mock_ratios_before',
                                                 'mock_ratios_after', 'tx_ratios', 'number_of_alleles'])
            self.df_mock_tx_snp_ratios = pd.concat([self.df_mock_tx_snp_ratios, curr_mock_df], ignore_index=True)

            # iterate over all possible snp
            phases = list(self._nuc_distribution_after.keys())
            for i in range(self._number_of_alleles):
                allele_window_list = list()
                # filter df for reads with the current snp and clean columns
                df_for_curr_allele = df.loc[df['snp_phase'] == phases[i]]
                # get the most relevant amplicon for reference from df
                amplicon = self._get_ref_amplicon(df_for_curr_allele, site_name)
                # TBD: think what to do if it does not the same - I think I fixed it with the function above
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
         :return: entropy: The entropy for this particular nucleotide base;
                  sorted_nuc_dict: A dictionary of the nucleotides' distribution
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
         :return: start: Start coordinate of the alignment;
                  end: End coordinate of the alignment
         """
        if not CTC:
            # alignment settings for not CTC (CTC = close to cut)
            # TBD: make as util
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
            # TBD: make as util
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
         :return: df: The df with some returned reads (that were able to be returned);
                  df_dropped: Reads that were not able to be returned
         """
        df_dropped_list = list()
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
                    # only rows that all snps have been found # TBD: check with Zohar if ok
                    num_reads_filtered_out += row['frequency']
                    df_dropped_list.append(row)
                    filtered_df = filtered_df.drop(i)
                    break
        # TBD: add to log
        df_dropped = pd.DataFrame(data=df_dropped_list, columns=list(filtered_df.columns))
        print(f'Length of dropped: {len(df_dropped)}')
        df = pd.concat([df, filtered_df])

        return df, df_dropped

    def _compute_lengths_distribution(self, sorted_df, possible_len):
        """
         Compute the distribution of the length of all the *aligned* reads in the df
         :param sorted_df: sorted df by frequency
         :param possible_len: list of all possible lengths in the df
         :return: lengths_ratio: The ratio of the largest length percentage;
                  len_dict: Dictionary of lengths percentages
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
         :return: df: return the same df as self._df however with additional columns regarding the snps;
                  SNP: True/False to indicate if SNP is exists or not
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
         :return: nuc_dict: normalized nuc dictionary distribution
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
        # making sure that the dictionary is sorted
        nuc_dict = dict(sorted(nuc_dict.items(), key=lambda item: item[1], reverse=True))
        # normalized the dictionary of the nucleotide distribution
        self._total_num_reads = sum(nuc_dict.values())
        for snp in nuc_dict.keys():
            nuc_dict[snp] /= self._total_num_reads
        return nuc_dict

    def _get_most_frq_cut_site(self, same_len_df):
        """
         Retrieve the most frequent cut-site to be the cut-site of the site
         :param: same_len_df: df of the site
         :return: the cut-site
         """
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
         :return: coverage_list: The list of coverage;
                  entropy_scores: The scores of all entropies of all number of potential alleles;
                  norm_scores_list: Score after normalization;
                  number_of_alleles: The optimum number of alleles
         """
        entropy_scores = list()
        norm_scores_list = list()
        coverage_list = list(self._nuc_distribution_before.values())

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

    def _mock_SNP_detection(self):
        """
        Find all snps and return df with the snp bases
        :param: self: Get some self variables
        :return: df: return the same df as self._df however with additional columns regarding the snps,
                SNP: True/False to indicate if SNP is exists or not
        """
        SNP = False
        false_df = False
        try:
            df = self._df
            df.loc[:, 'snp_nuc_type'] = [list() for _ in range(len(df.index))]  # create snp list for each row
            df = df.sort_values(['frequency'], ascending=False)  # sort according to frequency
            original_instances_number = sum(df['frequency'])  # count the number of original reads, TBD: delete / log

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

            # TBD: make "5" hyperparameter. but can't be too big because it cannot create directory with too long name
            if (len(self._snp_locus) > 0) and (len(self._snp_locus) < 5):
                SNP = True
                same_len_freq = sum(same_len_df['frequency'])
                # handle reads that are not at length of the self._consensus length
                reference_read = same_len_df.loc[list(same_len_df['frequency']).index(max(same_len_df['frequency'])),
                                                 'alignment_w_del']
                windows_list_for_realignment = self._get_windows(reference_read, cut_site)
                df_w_returned, df_dropped = self._return_reads_to_nuc_dist(
                    same_len_df, filtered_reads_diff_len, windows_list_for_realignment)

                # TBD ADD: df_dropped to the site dictionary as an output
                df_w_returned_freq = sum(df_w_returned['frequency'])
                df_dropped_freq = sum(df_dropped['frequency'])
                # TBD: add to log - {sum_read_after_adding-sum_read_before_adding} added during length processing

                # phasing the multi-snp into one phase instead of a list of snps
                df_w_returned['snp_phase'] = df_w_returned['snp_nuc_type'].apply(lambda x: ''.join(x))
                df_dropped['snp_phase'] = df_dropped['snp_nuc_type'].apply(lambda x: ''.join(x))
                # normalized the snp distribution # TBD: a little time-consuming. check
                self._nuc_distribution_before = self._norm_nuc_distribution(df_w_returned[['snp_phase', 'frequency']])
                # get the number of alleles
                _, _, _, self._number_of_alleles = self._get_num_alleles()

                # assign alleles that are not represented to alleles that do
                one_snp = False
                if len(self._snp_locus) == 1:
                    one_snp = True
                df_complete, filtered_due_align_freq = self._determine_allele_for_unknowns(df_w_returned, df_dropped,
                                                                  windows_list_for_realignment, one_snp)

                final_reads_number = sum(df_complete['frequency'])

                # normalizing again, after determine unknown reads' allele
                self._nuc_distribution_after = self._norm_nuc_distribution(df_complete[['snp_phase', 'frequency']])

                self.reads_drop_down.append([self._site_name, original_instances_number, same_len_freq,
                                             df_w_returned_freq, df_dropped_freq, filtered_due_align_freq,
                                             final_reads_number])

                return df_complete, SNP

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

    def _determine_allele_for_unknowns(self, df_w_returned, df_dropped, windows_list_for_realignment, one_snp):
        """
        """
        reads_to_be_filtered_list = list()
        allele_phases = list(self._nuc_distribution_before.keys())[:self._number_of_alleles]
        reads_with_alleles = df_w_returned[df_w_returned['snp_phase'].isin(allele_phases)]
        reads_with_no_alleles = df_w_returned[~df_w_returned['snp_phase'].isin(allele_phases)]
        if len(df_dropped) > 0:
            reads_with_no_alleles = pd.concat([reads_with_no_alleles, df_dropped])
        if one_snp:     # TBD: check if needed. Maybe redundant (966). It's NOT!!!
            for i, row in reads_with_no_alleles.iterrows():
                snp_type = random.choice(allele_phases)
                reads_with_no_alleles.at[i, 'snp_phase'] = snp_type
        else:
            windows_list = self._prepare_windows_for_alignment(windows_list_for_realignment, allele_phases)
            for i, row in reads_with_no_alleles.iterrows():
                if len(row['snp_phase']) == 0:
                    snp_type = random.choice(allele_phases)
                    reads_with_no_alleles.at[i, 'snp_phase'] = snp_type
                else:
                    seq = row['alignment_w_del']
                    # find the best allele to assign this read. return best index of allele list
                    best_index, score, _, _ = best_index_wrapper(seq, windows_list, self._snp_locus)
                    if score != None:
                        snp_type = allele_phases[best_index]
                        reads_with_no_alleles.at[i, 'snp_phase'] = snp_type
                    else:
                        # # reshaping row and appending
                        # row_to_append = pd.DataFrame(row.values.reshape(1, len(reads_with_no_alleles.columns))[0]).T
                        # row_to_append.rename(index={0: i}, inplace=True)
                        # row_to_append.columns = reads_with_no_alleles.columns
                        # # raise a numpy.VisibleDeprecationWarning error. canceled it the error
                        # reads_to_be_filtered = pd.concat([reads_to_be_filtered, row_to_append])
                        reads_to_be_filtered_list.append(row)
                        reads_with_no_alleles = reads_with_no_alleles.drop(i)

        # TBD: add to log the reads_to_be_filtered and output it as a file
        reads_to_be_filtered = pd.DataFrame(data=reads_to_be_filtered_list, columns=df_w_returned.columns)
        reads_with_alleles = pd.concat([reads_with_alleles, reads_with_no_alleles])

        return reads_with_alleles, sum(reads_to_be_filtered['frequency'])

    def _prepare_windows_for_alignment(self, windows_list_for_realignment, allele_phases):
        windows_list = list()
        for phase in allele_phases:
            temp_allele_list = list()
            for i, window in enumerate(windows_list_for_realignment):
                new_window = window[0].replace('N', phase[i])
                temp_allele_list.append((new_window, window[1], window[2], window[3]))
            windows_list.append(temp_allele_list)
        return windows_list


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
        self.reads_dropdown = list()

    # -------------------------------#
    ######### Public methods #########
    # -------------------------------#

    def run(self, ratios_df):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        original_sites = dict()
        for tx_site_name, tx_site_df in self._tx_df.items():
            # TBD: DELETE start
            print(tx_site_name)
            if tx_site_name == 'gINS11_FANCA_0029':
                print('ok')
            # TBD: DELETE end

            # if tx site has an allele site
            if tx_site_name in self._sites_to_be_alleles:
                temp_original_site_name = tx_site_name+'_Second'
                # prepare relevant information for execution
                self._new_tx_df[tx_site_name] = []
                # these modifications changing the tx_reads_d from the main
                tx_site_df['allele'] = None
                tx_site_df['is_random'] = False
                tx_site_df['is_filtered'] = False
                tx_site_df['alignment_score'] = math.nan
                # uncertain_reads = pd.DataFrame(columns=tx_site_df.columns)
                # reads_to_be_filtered = pd.DataFrame(columns=tx_site_df.columns)
                uncertain_reads_list = list()
                reads_to_be_filtered_list = list()
                # list of all relevant allele dfs for this site
                list_sites_of_allele = self._new_alleles[tx_site_name]
                windows = list()
                new_sites_name = list()
                # CTC_list = list()
                SNP_loci = list_sites_of_allele[0][2]
                # iterate over all possible alleles and add the window and the name of the allele site
                for site_allele in list_sites_of_allele:
                    windows.append(site_allele[3])
                    new_sites_name.append(site_allele[0])

                # iterate over all rows in df and assign to relevant allele
                for i, row in tx_site_df.iterrows():
                    # scores = dict() # TBD: delete
                    seq = row['alignment_w_del']
                    # find the best allele to assign this read. return best index of allele list
                    best_index, alignment_score, is_random, _ = best_index_wrapper(seq, windows, SNP_loci)
                    if alignment_score != None:
                        tx_site_df.loc[i, 'allele'] = new_sites_name[best_index]
                        # alignment_score = self._get_alignment_score(best_index, scores) # TBD: delete
                        tx_site_df.loc[i, 'alignment_score'] = alignment_score
                        # TBD: delete or put in log - check the random assignment
                        if is_random:
                            # # reshaping row and appending
                            # row_to_append = pd.DataFrame(row.values.reshape(1, len(uncertain_reads.columns))[0]).T
                            # row_to_append.rename(index={0: i}, inplace=True)
                            # row_to_append.columns = uncertain_reads.columns
                            # raise a numpy.VisibleDeprecationWarning error. canceled it the error
                            # row_to_append.loc[i, 'alignment_score'] = alignment_score
                            row['alignment_score'] = alignment_score
                            tx_site_df.loc[i, 'is_random'] = True
                            # uncertain_reads = pd.concat([uncertain_reads, row_to_append])
                            uncertain_reads_list.append(row)
                    else:
                        # # reshaping row and appending
                        # row_to_append = pd.DataFrame(row.values.reshape(1, len(reads_to_be_filtered.columns))[0]).T
                        # row_to_append.rename(index={0: i}, inplace=True)
                        # row_to_append.columns = reads_to_be_filtered.columns
                        # raise a numpy.VisibleDeprecationWarning error. canceled it the error
                        # row_to_append.loc[i, 'alignment_score'] = alignment_score
                        # reads_to_be_filtered = pd.concat([reads_to_be_filtered, row_to_append])
                        reads_to_be_filtered_list.append(row)
                        tx_site_df.loc[i, 'is_filtered'] = True
                        # tx_site_df = tx_site_df.drop(i) # TBD: is it deleting it from the original site as well?

                uncertain_reads = pd.DataFrame(data=uncertain_reads_list, columns=tx_site_df.columns)
                reads_to_be_filtered = pd.DataFrame(data=reads_to_be_filtered_list, columns=tx_site_df.columns)
                tx_reads_freq = sum(tx_site_df['frequency'])
                filtered_freq = sum(reads_to_be_filtered['frequency'])
                uncertain_freq = sum(uncertain_reads['frequency'])
                self.reads_dropdown.append([tx_site_name, tx_reads_freq, filtered_freq, uncertain_freq])

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
                    tx_site_df_temp = tx_site_df_temp.drop(labels=['allele', 'alignment_score'], axis=1)  # TBD: maybe i should add 'is_random'
                    self._new_tx_df[tx_site_name].append([new_site, tx_site_df_temp])
                original_sites[tx_site_name] = tx_site_df   # TBD: maybe delete
                # TBD: insert to log:
                self.uncertain_reads[tx_site_name] = [sum(uncertain_reads['frequency']),
                                                      sum(tx_site_df['frequency'])]

                # normalize the dict
                try:
                    sum_all_reads = sum(ratios_dict.values())
                    for _i in ratios_dict.keys():
                        ratios_dict[_i] /= sum_all_reads
                except:
                    print(f'problem with {tx_site_name}')

                # append to df - TBD: return later
                ratios_df.at[list(ratios_df['site_name']).index(tx_site_name), 'tx_ratios'] = ratios_dict

        return self._new_tx_df, original_sites, round(self._sites_score, 3), ratios_df

    # -------------------------------#
    ######### Private methods ########
    # -------------------------------#

    # TBD: delete
    # def _get_alignment_score(self, indx, scores):
    #     sum_all = 0
    #     for lst in scores.values():
    #         sum_all += lst[indx]
    #     return sum_all / len(scores)


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
                new_amplicon = site_allele[4]
                # SNP_window = site_allele[3]
                # loci = site_allele[2]
                # new_amplicon = self.create_new_amplicon(new_site_name,original_site_name,SNP_window,loci)
                new_row_for_ref_df = self._ref_df[self._ref_df['Site Name'] == original_site_name]
                new_row_for_ref_df = new_row_for_ref_df.rename(index={original_site_name: new_site_name})
                new_row_for_ref_df.loc[new_site_name, 'AmpliconReference'] = new_amplicon
                new_row_for_ref_df.loc[new_site_name, 'Site Name'] = new_site_name
                self._ref_df = pd.concat([self._ref_df, new_row_for_ref_df])    # TBD: maybe replace concat with append to list like I did in previous cases
        '''add PAM coordinates and gRNA coordinates to df'''
        # TBD: WHY to get the PAM?? cannot remember
        self._get_PAM_site()

        return self._ref_df

    # -------------------------------#
    ######### Private methods #######
    # -------------------------------#

    # TBD: delete
    # def create_new_amplicon(self, new_site_name, original_site_name, SNP_window, loci):
    #     curr_amplicon = str(self._ref_df[self._ref_df['Site Name'] == original_site_name]['AmpliconReference'][0])
    #     SNP_locus, SNP_score = get_snp_locus_and_score(curr_amplicon, SNP_window)
    #     new_amplicon = curr_amplicon[:SNP_locus] + new_site_name[-1] + curr_amplicon[SNP_locus + 1:]
    #     '''TBD: print to be deleted'''
    #     print(f'The gap between SNP and new SNP locus is: {loci} - {SNP_locus} = {loci - SNP_locus}')

    # return new_amplicon

    def _get_PAM_site(self):
        """
        Set the gRNA coordinate and PAM coordinate for each site
        :param:
        :return: assign values directly to df
        """
        self._ref_df['PAM_window'] = None
        self._ref_df['grna_window'] = None

        for i, row in self._ref_df.iterrows():
            cut_site = row['cut-site']
            if row['sgRNA_reversed']:
                PAM = [cut_site - 6, cut_site - 4]
                grna = [cut_site - 3, cut_site + 16]
            elif not row['sgRNA_reversed']:
                PAM = [cut_site + 3, cut_site + 5]
                grna = [cut_site - 17, cut_site + 2]

            self._ref_df.at[i, 'PAM_window'] = PAM
            self._ref_df.at[i, 'grna_window'] = grna


# -------------------------------#
######## Global Functions ########
# -------------------------------#
# TBD: make util parameter
searching_bounds = 30  # TBD: make util parameter
half_window_len = 10  # TBD: make util parameter
local_aligner = Align.PairwiseAligner()
local_aligner.mode = 'local'
local_aligner.match = 5
local_aligner.mismatch = -4
local_aligner.open_gap_score = -10
local_aligner.extend_gap_score = 0
local_aligner.target_end_gap_score = 0.0
local_aligner.query_end_gap_score = 0.0

MAX_MM = 5  # maximum mismatches allowed when aligning guide to amplicon (both mock and tx) # TBD: in util


def get_most_freq_val(lst):
    """
    Get the most frequent values from a list
    :param: lst: list of values
    :return: the most frequent values (can be many)
    """
    # using df to get the desired output
    df = pd.DataFrame({'Number': lst})
    df1 = pd.DataFrame(data=df['Number'].value_counts())
    df1['Count'] = df1['Number']
    df1['Number'] = df1.index
    df1 = df1.reset_index(drop=True)

    return list(df1[df1['Count'] == df1.Count.max()]['Number'])


def count_diff(seq_A, seq_B):
    """
    Count the differences between 2 sequences
    :param: seq_A: first sequence
    :param: seq_B: second sequence
    :return: counter: the number of mismatches at the same positions
    """
    counter = 0
    for i, j in zip(seq_A, seq_B):
        if i != j:
            counter += 1
    return counter


def calc_alignments_scores(curr_amplicon, SNP_window_information, SNP_loci, CTC_list):
    """
    Calculates the best alignment score between each window to a given amplicon
    :param: curr_amplicon: The amplicon to be aligned to
    :param: SNP_window_information: list of SNP windows with additional information
    :param: SNP_loci: list of all snps of this site
    :param: CTC_list: list of True/False regarding the snp close to site or not
    :return: total_scores: list of lists - best score for each snp for each allele option.
                           Inner list - scores of all alleles in a certain snp
    """
    total_scores = list()
    total_i = list()

    # iterate over different snp
    for i, snp_option in enumerate(SNP_window_information[0]):
        temp_scores = list()
        smallest_i_list = list()
        SNP_locus = SNP_loci[i]
        # setting a smaller part of amplicon to align to. Save more computing time
        if SNP_locus - searching_bounds < 0:
            relevant_read = curr_amplicon[:searching_bounds * 2]
        elif SNP_locus + searching_bounds > len(curr_amplicon):
            relevant_read = curr_amplicon[-searching_bounds * 2:]
        else:
            relevant_read = curr_amplicon[SNP_locus - searching_bounds:SNP_locus + searching_bounds]

        # iterate over all alleles
        for j in range(len(SNP_window_information)):

            SNP_window = SNP_window_information[j][i][0]
            CTC = CTC_list[i]
            add_nb_before = SNP_window_information[j][i][2]
            add_nb_after = SNP_window_information[j][i][3]

            smallest_gap = np.inf
            i_smallest = None

            # if the snp not close to the cut site
            if not CTC:
                for t in range(len(relevant_read) - (len(SNP_window)) + 1):
                    window = relevant_read[t: t + len(SNP_window)]
                    curr_gap_score = count_diff(SNP_window, window)
                    if curr_gap_score < smallest_gap:
                        smallest_gap = curr_gap_score
                        i_smallest = t
                temp_scores.append(smallest_gap)
                smallest_i_list.append(i_smallest)

            # if the snp is close to the cut site
            elif CTC:
                alignment_for_norm = local_aligner.align(SNP_window, SNP_window)
                max_CTC_score = alignment_for_norm.score
                alignment = local_aligner.align(relevant_read, SNP_window)
                alignment_score = alignment.score
                norm_score = - alignment_score / max_CTC_score
                temp_scores.append(norm_score)

        total_scores.append(temp_scores)
        total_i.append(smallest_i_list)

    return total_scores


def determine_best_idx(curr_amplicon, SNP_window_information, SNP_loci, CTC_list, scores):
    """
    Determine the index with the best score out of list of lists
    :param: scores: list of lists with scores as values
    :param: CTC_list: list of Close To Cut with relative to each snp
    :param: curr_amplicon: The amplicon to be aligned to
    :param: SNP_window_information: list of SNP windows with additional information
    :param: SNP_loci: list of all snps of this site
    :return: returns the best index (or False) and the index_list
    """
    index_list = list()
    CTC_alignment = False
    is_random = False
    # iterate over all list. each list is the scores of all alleles in a given snp
    for score in scores:
        # set the minimum score in the list and find all indexes that comply the minimum score
        min_score = min(score)
        all_min_idx = [idx for idx, val in enumerate(score) if val == min_score]
        index_list.append(all_min_idx)
    # get the best indexes across all lists
    best_indexes = list(set(index_list[0]).intersection(*index_list))
    # if the best_indexes contains only one index - return it
    if len(best_indexes) == 1:
        best_index = best_indexes[0]
    # else - if there are several "best index":
    else:
        # if it is only one snp
        if len(scores) == 1:    # TBD: check if needed. Maybe redundant (582) No it's not. important for Tx
            is_random = True
            best_index = random.choice(index_list[0])
            if CTC_list[0]:
                CTC_alignment = True
        else:
            # flatten all lists to one list of best indexes
            flat_best_index_list = [item for sublist in index_list for item in sublist]
            # get the indexes that repeated the most out of the flatted list # TBD: ask Zohar if this good method
            new_best_indexes = get_most_freq_val(flat_best_index_list)
            # if there is an allele that is more dominant in the sense of fitting to window - get it.
            if len(new_best_indexes) == 1:
                best_index = new_best_indexes[0]
            else:
                # create 2 lists - one to contain all scores, and
                # one to contain all scores but without scores of snps that close to cut-site
                sub_scores_w_CTC = list()
                sub_scores_wo_CTC = list()
                # iterate over all scores
                for j, score in enumerate(scores):
                    # append to the lists the scores in the new_best_indexes
                    sub_scores_w_CTC.append([x for i, x in enumerate(score) if i in new_best_indexes])
                    if not CTC_list[j]:
                        sub_scores_wo_CTC.append([x for i, x in enumerate(score) if i in new_best_indexes])
                sums = np.sum(sub_scores_wo_CTC, axis=0)
                # if all sums are equal - we need to try different approach
                if np.count_nonzero(np.array(sums) == min(sums)) > 1:
                    # if there is a CTC between snps
                    if (sub_scores_w_CTC != sub_scores_wo_CTC) and (len(sub_scores_wo_CTC) > 0):
                        CTC_alignment = True
                        # consider all snps CTC and calculate from the beginning the scores
                        new_CTC_list = [True] * len(CTC_list)
                        scores_new = calc_alignments_scores(curr_amplicon, SNP_window_information, SNP_loci, new_CTC_list)
                        scores = [x for i, x in enumerate(scores_new) if i in new_best_indexes]
                        new_sums = np.sum(scores, axis=0)
                        # if all sums are equal - choose randomly out of the option of new_best_indexes
                        if np.count_nonzero(np.array(new_sums) == min(new_sums)) > 1:
                            is_random = True
                            best_index = random.choice(new_best_indexes)
                        # else - we have a better index in a sense of scoring, thus we will choose it.
                        else:
                            best_index = new_best_indexes[np.argmin(new_sums)]
                    # else - there is no CTC among snps,
                    # thus we don't have more calculation to make, so we will pick index randomly
                    else:
                        is_random = True
                        new_min_idx = [idx for idx, val in enumerate(sums) if val == min(sums)]
                        best_index = new_best_indexes[random.choice(new_min_idx)]
                # else - we have a better index in a sense of scoring, thus we will choose it.
                else:
                    best_index = new_best_indexes[np.argmin(sums)]

    alignment_score = np.mean(scores, axis=0)[best_index]
    if CTC_alignment:
        MAX_CTC_MM = - local_aligner.match * (half_window_len * 2 + 1 - MAX_MM) \
                     / (local_aligner.match * (half_window_len * 2 + 1))
        if alignment_score > MAX_CTC_MM:
            alignment_score = None
    else:
        if alignment_score > MAX_MM:
            alignment_score = None

    return best_index, alignment_score, is_random, index_list


def best_index_wrapper(curr_amplicon, SNP_window_information, SNP_loci):
    """
    wrapper of the two functions that determine the best_index
    :param: curr_amplicon: The amplicon to be aligned to
    :param: SNP_window_information: list of SNP windows with additional information
    :param: SNP_loci: list of all snps of this site
    :param: CTC_list: list of True/False regarding the snp close to site or not
    :return: df: the best index of allele to be set, alignment score, if it was picked by random, the whole list
    """
    # add to list the True/False for each CTC window
    CTC_list = list()       # TBD: put CTC into a function outside here and outside every row iteration
    for window_info in SNP_window_information[0]:
        CTC_list.append(window_info[1])
    scores = calc_alignments_scores(curr_amplicon, SNP_window_information, SNP_loci, CTC_list)
    best_index, alignment_score, is_random, index_list = determine_best_idx(curr_amplicon, SNP_window_information,
                                                                            SNP_loci, CTC_list, scores)
    return best_index, alignment_score, is_random, index_list


# TBD: delete
# def get_snp_locus_and_score(curr_amplicon, SNP_window_information, SNP_locus):
#     SNP_window = SNP_window_information[0]
#     CTC = SNP_window_information[1]
#     add_nb_before = SNP_window_information[2]
#     add_nb_after = SNP_window_information[3]
#     smallest_gap = np.inf
#     i_smallest = None
#     searching_bounds = 30  # TBD: make util parameter
#     half_window_len = 10  # TBD: make util parameter
#     if SNP_locus - searching_bounds < 0:
#         relevant_read = curr_amplicon[:searching_bounds * 2]
#     elif SNP_locus + searching_bounds > len(curr_amplicon):
#         relevant_read = curr_amplicon[-searching_bounds * 2:]
#     else:
#         relevant_read = curr_amplicon[SNP_locus - searching_bounds:SNP_locus + searching_bounds]
#
#     for i in range(len(relevant_read) - (len(SNP_window)) + 1):
#         window = relevant_read[i: i + len(SNP_window)]
#         curr_gap_score = count_diff(SNP_window, window)
#         if curr_gap_score < smallest_gap:
#             smallest_gap = curr_gap_score
#             i_smallest = i
#     # TBD CHANE: change 10 to self._half_window_len which will be in utils
#     return i_smallest + half_window_len + add_nb_before - add_nb_after, smallest_gap


# TBD: why not in use?
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
