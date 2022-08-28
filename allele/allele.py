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
    def __init__(self, ratios):
        self._site_name = None
        self._df = None
        # TBD: make as a util parameter
        self._ratios = ratios
        self._new_allels = dict()
        self._consensus_len = None
        # TBD change: ratio to filter out from mock-snp-detection due to different length of reads:
        self._length_ratio = 0.3
        # TBD change: one side length of window opening: keep it even number.
        # TBD: put it in utils and import it from there
        self._half_window_len = 10
        # TBD change: set df for the ratios in mock vs. tx
        self._df_mock_tx_snp_ratios = pd.DataFrame(columns=['site_name', 'mock_ratios', 'tx_ratios'])

    # -------------------------------#
    ######### Public methods #########
    # -------------------------------#

    def run(self,site_name, df):
        self._site_name = site_name
        self._df = df
        self._snp_locus = []
        self._total_num_reads = 0
        self._nuc_distribution = None
        self._window = None

        # TBD DELETE
        if site_name == 'gINS11_FANCA_60':
            print('ok')
        df, SNP = self.mock_SNP_detection()

        # if the function finds SNP's
        if SNP:
            df['snp_nuc_type'] = df['snp_nuc_type'].apply(lambda x: ''.join(x))
            # insert ratios to the self variable
            curr_mock_df = pd.DataFrame(data=[[self._site_name, self._nuc_distribution, None]],
                                        columns=['site_name', 'mock_ratios', 'tx_ratios'])
            self._df_mock_tx_snp_ratios = pd.concat([self._df_mock_tx_snp_ratios, curr_mock_df], ignore_index=True)

            # iterate over all possible snp
            for _key, _num in self._nuc_distribution.items():
                # TBD change: change 0.20 to "number of alleles" that will be calculated in the mock_SNP_detection
                if _num >= 0.20:
                    # filter df for reads with the current snp and clean columns
                    df_for_curr_allele = df.loc[df['snp_nuc_type'] == _key]
                    df_for_curr_allele = df_for_curr_allele.drop(labels='snp_nuc_type', axis=1)
                    df_for_curr_allele = df_for_curr_allele.drop(labels='len', axis=1)
                    # prepare the window_search and the new_site_name
                    allele_window = self._window[:self._half_window_len] + _key + self._window[self._half_window_len + 1:]
                    _new_name = self._site_name + '_' + str(self._snp_locus[0]) + '_' + _key

                    # add list of:(new name, filtered df, snp position, window)
                    if self._site_name in self._new_allels.keys():
                        self._new_allels[self._site_name].append([
                            _new_name,
                            df_for_curr_allele,
                            self._snp_locus[0],
                            allele_window])
                    else:
                        self._new_allels[self._site_name] = [[
                            _new_name,
                            df_for_curr_allele,
                            self._snp_locus[0],
                            allele_window]]

        return self._new_allels

    #-------------------------------#
    ######### Private methods #######
    #-------------------------------#

    def calc_entropy(self,temp_df):
        """
         Calculates entropy of the passed `pd.Series`
         :param df: One nucleotide base through all reads in one mock site, along with frequancy of each read
         :return: The entropy for this particular nucleotide base & a dictionary of the nucleodites' ditribution
         """
        # computing unique nuc bases
        df = temp_df.set_axis(['base', 'freq'], axis=1, inplace=False)
        nuc_bases = set(df.iloc[:, 0])
        nuc_dict = {}
        # Not eliminating '-' sign from the entropy calculation, as it can act as nuc
        # nuc_bases.discard('-')
        # computing nuc bases distribution
        for nuc in nuc_bases:
            nuc_dict[nuc] = df.loc[df['base'] == nuc, 'freq'].sum()
        sorted_nuc_dict = dict(sorted(nuc_dict.items(), key=lambda item: item[1], reverse=True))
        # take only the 2 largests items
        dict_for_entropy = {k: sorted_nuc_dict[k] for k in list(sorted_nuc_dict)[:2]}
        # calc entropy
        entropy = scipy.stats.entropy(pd.Series(dict_for_entropy), base=2)  # get entropy from counts, log base 2

        return entropy, sorted_nuc_dict

    def return_reads_to_nuc_dist(self, df, filtered_df, reference_read, snp_locus, dict_dist_nuc):

        # alignment_settings
        local_aligner = Align.PairwiseAligner()
        local_aligner.mode = 'local'
        local_aligner.match = 5
        local_aligner.mismatch = 1
        local_aligner.open_gap_score = -100
        local_aligner.extend_gap_score = -100
        local_aligner.target_end_gap_score = 0.0
        local_aligner.query_end_gap_score = 0.0
        df_dropped = pd.DataFrame(columns=list(filtered_df.columns))
        num_reads_filtered_out = 0
        # settings end

        window_search = reference_read[snp_locus-self._half_window_len:snp_locus] \
                        + 'N' + \
                        reference_read[snp_locus:snp_locus+self._half_window_len]

        for i, row in filtered_df.iterrows():
            read = row['alignment_w_del']
            alignment = local_aligner.align(read, window_search)
            [read_aligned, matches, window_aligned, _] = format(alignment[0]).split("\n")
            # finding positions of start and end in the original read
            start1 = matches.find('|')
            start2 = matches.find('.')
            end1 = matches.rfind('|')
            end2 = matches.rfind('.')
            start = min(start1, start2)
            end = max(end1, end2) + 1
            if end-start == len(window_search):
                snp_nb = read[start:end][self._half_window_len]
                filtered_df.at[i, 'snp_nuc_type'].append(snp_nb)
                if snp_nb in dict_dist_nuc.keys():
                    dict_dist_nuc[snp_nb] += row['frequency']
                else:
                    dict_dist_nuc[snp_nb] = row['frequency']
            else:
                num_reads_filtered_out += row['frequency']
                # TBD: add to log
                df_dropped = pd.concat([df_dropped, pd.DataFrame(row).transpose()])
                filtered_df.drop(i)


        df = pd.concat([df, filtered_df])

        return dict_dist_nuc, df, df_dropped



    def mock_SNP_detection(self):
        """
         Calculates entropy of the passed `pd.Series`
         :param name: One nucleotide base through all reads in one mock site, along with frequancy of each read
         :param df: One nucleotide base through all reads in one mock site, along with frequancy of each read
         :param df: One nucleotide base through all reads in one mock site, along with frequancy of each read
         :return: The entropy for this particular nucleotide base & a dictionary of the nucleodites' ditribution
         """
        SNP = False
        try:
            # relevant_columns = ['alignment_w_del','frequency']
            # df = self._df[relevant_columns]
            df = self._df
            # rename columns
            # df.columns = ['alignment', 'frequency']
            df.loc[:, 'snp_nuc_type'] = [list() for x in range(len(df.index))]
            sorted_df = df.sort_values(['frequency'], ascending=False)   # sort according to frequency
            original_instance_number = sum(sorted_df['frequency'])       # count the number of original reads

            # adding len of reads to df
            len_reads = list(sorted_df['alignment_w_del'].str.len())
            sorted_df['len'] = len_reads

            # compute the distribution of the length of the reads
            possible_len = set(len_reads)
            len_dict = {}
            for i_len in possible_len:
                len_dict[i_len] = sorted_df.loc[sorted_df['len'] == i_len, 'frequency'].sum()
            sum_len = sum(len_dict.values())
            values_len = list(len_dict.values())
            percent_len = [x / sum_len for x in values_len]
            dist_len = np.sort(percent_len)[::-1]

            # TBD confirm: if the length of the reads differ and have sort-of uniform distribution
            # if the highest freq of length is smaller than 70% than filter out all 30%
            umbigiuos_reads_length = False
            if dist_len[0] < (1 - self._length_ratio):
                umbigiuos_reads_length = True
                # TBD ADD: add to log - note that filter a {percentage of reads} from this site
                # TBD ADD: create a file of umbigiuos reads in the site directory


            # calc the majority number of reads and filter Dataframe based on it
            self._consensus_len = max(len_dict, key=len_dict.get)
            mask = sorted_df['len'] == self._consensus_len
            same_len_df = sorted_df.loc[mask]
            filtered_reads_diff_len = sorted_df.loc[~mask]

            # TBD: DELETE!
            # compute the current number of reads and the gap
            # current_instance_number = sum(same_len_df['frequency'])
            # gap_reads = original_instance_number - current_instance_number
            #
            # '''TBD: What to do if too many reads were filtered??'''
            # if gap_reads/original_instance_number > 0.3:
            #     print(f'Too many reads were filtered({gap_reads/original_instance_number})')
            #TBD: END DELETE


            # obtain entropys for the site run
            local_score = []

            # split the DataFrame into separated columns
            parsed_df = same_len_df['alignment_w_del'].apply(lambda x: pd.Series(list(x)))
            freq_list = same_len_df['frequency']

            # set the lower decision bound for significant entropy
            lower_bound_score = 0
            for p in self._ratios:
                # TBD: change log2 to hyperparameter
                lower_bound_score -= p * np.log2(p)

            list_of_dist_dic = []

            # iterate over all columns, calc entropy and append to local score
            for i in range(len(parsed_df.columns)):
                temp_df = pd.concat([parsed_df[i], freq_list], axis=1, join='inner')
                temp_score, dict_dist_nuc = self.calc_entropy(temp_df)
                local_score.append(temp_score)
                if temp_score >= lower_bound_score:
                    self._snp_locus.append(i)
                    list_of_dist_dic.append(dict_dist_nuc)
                else:
                    continue

            for i_row, row in same_len_df.iterrows():
                for snp_loc in self._snp_locus:
                    same_len_df.at[i_row, 'snp_nuc_type'].append(str(row['alignment_w_del'][snp_loc]))

            # TBD EXTEND: handle return of reads from filtered_df only for one snp for now:
            reference_read = same_len_df.loc[list(same_len_df['frequency']).index(max(same_len_df['frequency'])), 'alignment_w_del']
            sum_read_before_adding = sum(dict_dist_nuc.values())
            for i, snp_loc in enumerate(self._snp_locus):
                dict_dist_nuc, df_full_w_returned_reads, df_dropped = self.return_reads_to_nuc_dist(
                    same_len_df, filtered_reads_diff_len, reference_read, snp_loc, list_of_dist_dic[i]
                )
            # TBD ADD: df_dropped to the site dictionary
            sum_read_after_adding = sum(dict_dist_nuc.values())
            # TBD: add to log - {sum_read_after_adding-sum_read_before_adding} added during length processing

            if len(self._snp_locus) == 1:
                nuc_dict = list_of_dist_dic[0]
                self._total_num_reads = sum(nuc_dict.values())
                for _ in nuc_dict.keys():
                    nuc_dict[_] /= self._total_num_reads

                self._nuc_distribution = nuc_dict
                '''open a nucleotide window for future searching in tx reads'''
                window_from_this_read = df['alignment_w_del'][0]
                self._window = window_from_this_read[self._snp_locus[0]-self._half_window_len : self._snp_locus[0]+self._half_window_len+1]

                return df_full_w_returned_reads, True

            elif len(self._snp_locus) > 1:
                parsed_df['combined'] = parsed_df[self._snp_locus].values.sum(axis=1)
                SNP_combined_df = pd.concat([parsed_df['combined'], freq_list], axis=1, join='inner')

                # calc entropy again
                temp_loci_score, dict_dist_loci = self.calc_entropy(SNP_combined_df)
                local_score.append(temp_loci_score)
                if temp_loci_score >= lower_bound_score:
                    list_of_dist_dic.append(dict_dist_loci)

                    nuc_dict = dict_dist_loci
                    self._total_num_reads = sum(nuc_dict.values())
                    for _ in nuc_dict.keys():
                        nuc_dict[_] /= self._total_num_reads
                    self._nuc_distribution = nuc_dict

                    '''open a nucleotide window for future searching in tx reads'''
                    first_snp = self._snp_locus[0]
                    last_snp = self._snp_locus[-1]
                    # if last_snp - first_snp < 20:
                    #
                    # window_from_this_read = df['alignment'][0]
                    # self._window = window_from_this_read[self._snp_locus[0] - self._half_window_len: self._snp_locus[0] + self._half_window_len+1]

                    return False, False
                else:
                    return False, False
            else:
                return False, False
        except:
            return False, False


class AlleleForTx:
    """
    Class to handle the allelity case of the mock
    """
    def __init__(self):
        self._tx_df = None
        self._new_allels = None
        self._sites_to_be_alleles = []
        self._new_tx_df = dict()

    # -------------------------------#
    ######### Public methods #########
    # -------------------------------#

    def run(self,tx_df, new_allels, ratios_df):
        self._tx_df = tx_df
        self._new_allels = new_allels
        self._sites_score = pd.DataFrame(columns=['site_name', 'avg_score'])

        # create list of sites to be handled
        self._sites_to_be_alleles = list(self._new_allels.keys())

        for tx_site_name, tx_site_df in self._tx_df.items():
            if tx_site_name in self._sites_to_be_alleles:  # if tx site an allele site
                self._new_tx_df[tx_site_name] = []
                tx_site_df['allele'] = None
                tx_site_df['alignment_score'] = math.nan
                uncertain_reads = pd.DataFrame(columns=tx_site_df.columns)
                list_sites_of_allele = self._new_allels[tx_site_name]  # list of all relevant allele dfs for this site
                windows = list()
                new_sites_name = list()
                for site_allele in list_sites_of_allele:   # iterate over all possible alleles
                    windows.append(site_allele[3])         # add the window to list
                    new_sites_name.append(site_allele[0])  # add the new name to list
                for i, row in tx_site_df.iterrows():       # iterate over all rows in df and assign to relevant allele
                    scores = []
                    seq = row['alignment_w_del']
                    for SNP_window in windows:
                        _, curr_score = get_snp_locus_and_score(seq, SNP_window)
                        scores.append(curr_score)
                    # check if scores are equal, and if so, append to different df
                    if len(set(scores)) == 1:
                        # reshaping row and appending
                        row_to_append = pd.DataFrame(row.values.reshape(1, len(tx_site_df.columns))[0]).T
                        row_to_append.rename(index={0: i}, inplace=True)
                        row_to_append.columns = tx_site_df.columns
                        row_to_append.loc[i, 'alignment_score'] = np.min(scores)
                        uncertain_reads = pd.concat([uncertain_reads, row_to_append])
                    else:
                        tx_site_df.loc[i,'allele'] = new_sites_name[np.argmin(scores)]
                        tx_site_df.loc[i, 'alignment_score'] = np.min(scores)

                ratios_dict = dict()
                # set the new df for the tx allele
                for new_site in new_sites_name:
                    # get the tx ratios for this site
                    site_snp = new_site[-1]
                    tx_site_df_temp = tx_site_df.loc[tx_site_df['allele'] == new_site]
                    sum_reads = sum(list(tx_site_df_temp['frequency']))
                    ratios_dict[site_snp] = sum_reads
                    # calculate the alignment average score. If less than one [TBD]
                    avg_score = tx_site_df_temp['alignment_score'].mean(axis=0, skipna=True)
                    self._sites_score = self._sites_score.append({'site_name': new_site, 'avg_score':avg_score}, ignore_index=True)
                    tx_site_df_temp = tx_site_df_temp.drop(labels='allele', axis=1)
                    tx_site_df_temp = tx_site_df_temp.drop(labels='alignment_score', axis=1)
                    self._new_tx_df[tx_site_name].append([new_site,tx_site_df_temp])

                # return uncertain_reads back [TBD] - should it be after or before the tx_ratios?
                # bring_back = 2
                # if bring_back == 1:
                #     # first method - by the ratio of the mock
                #     self.merge_back_uncertain_read1(tx_site_name, uncertain_reads, ratios_df)
                # if bring_back == 2:
                #     # second method - return by aligning to the consensus in each site
                #     self.merge_back_uncertain_read2(tx_site_name, uncertain_reads)

                # normalize the dict
                sum_all_reads = sum(ratios_dict.values())
                for _i in ratios_dict.keys():
                    ratios_dict[_i] /= sum_all_reads

                # append to df
                ratios_df.at[list(ratios_df['site_name']).index(tx_site_name), 'tx_ratios'] = ratios_dict

        return self._new_tx_df, round(self._sites_score, 3), ratios_df

    #-------------------------------#
    ######### Private methods #######
    #-------------------------------#

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
                new_amplicon = self.create_new_amplicon(new_site_name,original_site_name,SNP_window,loci)
                new_row_for_ref_df = self._ref_df[self._ref_df['Site Name']==original_site_name]
                new_row_for_ref_df = new_row_for_ref_df.rename(index={original_site_name:new_site_name})
                new_row_for_ref_df.loc[new_site_name,'AmpliconReference'] = new_amplicon
                new_row_for_ref_df.loc[new_site_name,'Site Name'] = new_site_name
                self._ref_df = pd.concat([self._ref_df, new_row_for_ref_df])
        '''add PAM coordinates and grna coordinates to df'''
        self.get_PAM_site()

        return self._ref_df

    #-------------------------------#
    ######### Private methods #######
    #-------------------------------#

    def create_new_amplicon(self, new_site_name, original_site_name, SNP_window, loci):
        curr_amplicon = str(self._ref_df[self._ref_df['Site Name']==original_site_name]['AmpliconReference'][0])
        SNP_locus, SNP_score = get_snp_locus_and_score(curr_amplicon,SNP_window)
        new_amplicon = curr_amplicon[:SNP_locus] + new_site_name[-1] + curr_amplicon[SNP_locus+1:]
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
    for i in range(len(curr_amplicon) - (len(SNP_window)-1)):
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

























