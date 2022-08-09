import numpy as np
import pandas as pd
from scipy.stats import entropy
import scipy.stats

class AlleleForMock:
    """
    Class to handle the allelity case of the mock
    """
    def __init__(self, ratios):
        self._site_name = None
        self._df = None
        self._ratios = ratios
        self._new_allels = dict()

    # -------------------------------#
    ######### Public methods #########
    # -------------------------------#

    def run(self,site_name, df):
        self._site_name = site_name
        self._df = df
        self._snp_locus = list()
        self._total_num_reads = 0
        self._nuc_distribution = None
        self._window = None
        self._snp_in_window = list()

        if self.mock_SNP_detection(): # if the function find ONE SNP ONLY
            df = self._df
            df['snp_nuc_type'] = df.alignment_w_del.str[self._snp_locus[0]]
            for _key,_num in self._nuc_distribution.items():
                if _num > 0.25:
                    allele_window = self._window[:10] + _key + self._window[11:]
                    df_for_curr_allele = df.loc[df['snp_nuc_type'] == _key]
                    df_for_curr_allele = df_for_curr_allele.drop(labels='snp_nuc_type', axis=1)
                    '''the new name equals to: original name, window search, new df'''
                    _new_name = self._site_name + '_' + str(self._snp_locus[0]) + '_' + _key
                    if self._site_name in self._new_allels.keys():
                        self._new_allels[self._site_name].append((
                            _new_name,
                            df_for_curr_allele,
                            self._snp_locus[0],
                            allele_window))
                    else:
                        self._new_allels[self._site_name] = [(
                            _new_name,
                            df_for_curr_allele,
                            self._snp_locus[0],
                            allele_window)]


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
        entropy = scipy.stats.entropy(pd.Series(nuc_dict), base=2)  # get entropy from counts, log base 2

        sorted_nuc_dict = dict(sorted(nuc_dict.items(), key=lambda item: item[1]))
        return entropy, sorted_nuc_dict

    def mock_SNP_detection(self):
        """
         Calculates entropy of the passed `pd.Series`
         :param name: One nucleotide base through all reads in one mock site, along with frequancy of each read
         :param df: One nucleotide base through all reads in one mock site, along with frequancy of each read
         :param df: One nucleotide base through all reads in one mock site, along with frequancy of each read
         :return: The entropy for this particular nucleotide base & a dictionary of the nucleodites' ditribution
         """
        try:
            relevant_columns = ['alignment_w_del','frequency']
            df = self._df[relevant_columns]
            # rename columns
            df.columns = ['alignment', 'frequency']

            # sort according to frequency
            sorted_df = df.sort_values(['frequency'], ascending=False)

            # count the number of original reads
            original_instance_number = sum(sorted_df['frequency'])

            # adding len of reads to df
            len_reads = list(sorted_df['alignment'].str.len())
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

            # if the length of the reads differ and have sort-of uniform distribution - thus there are different alleles
            '''TBD: What to do in case of many unbalanced reads?!!!!'''
            # if (len(dist_len) > 1) and (dist_len[1] > 0.3) and (dist_len[0] + dist_len[1] > 0.95):
            #     print(f'This case is diploid allele')


            # calc the majority number of reads and filter Dataframe based on it
            majority = max(len_dict, key=len_dict.get)
            mask = sorted_df['len'] == majority
            sorted_df = sorted_df.loc[mask]

            # compute the current number of reads and the gap
            current_instance_number = sum(sorted_df['frequency'])
            gap_reads = original_instance_number - current_instance_number

            '''TBD: What to do if too many reads were filtered??'''
            if gap_reads/original_instance_number > 0.3:
                print(f'Too many reads were filtered({gap_reads/original_instance_number})')

            # obtain entropys for the site run
            local_score = []

            # split the DataFrame into separated columns
            parsed_df = sorted_df['alignment'].apply(lambda x: pd.Series(list(x)))
            freq_list = sorted_df['frequency']

            # set the lower decision bound for significant entropy
            lower_bound_score = 0
            for p in self._ratios:
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

            if len(self._snp_locus) == 1:
                nuc_dict = list_of_dist_dic[0]
                self._total_num_reads = sum(nuc_dict.values())
                for _ in nuc_dict.keys():
                    nuc_dict[_] /= self._total_num_reads

                self._nuc_distribution = nuc_dict
                '''open a nucleotide window for future searching in tx reads'''
                window_from_this_read = df['alignment'][0]
                self._window = window_from_this_read[self._snp_locus[0]-10 : self._snp_locus[0]+11]
                self._snp_in_window.append(int(len(self._window)//2))

                return True

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
                    if last_snp - first_snp < 20:
                        window_from_this_read = df['alignment'][0]
                        self._window = window_from_this_read[self._snp_locus[0] - 10: self._snp_locus[0] + 11]

                    return True
                else:
                    return False

            else:
                return False

        except:
            return False


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

    def run(self,tx_df, new_allels):
        self._tx_df = tx_df
        self._new_allels = new_allels

        # create list of sites to be handled
        self._sites_to_be_alleles = list(self._new_allels.keys())

        for tx_site_name, tx_site_df in self._tx_df.items():
            if tx_site_name in self._sites_to_be_alleles:
                tx_site_df['allele'] = None
                list_sites_of_allele = self._new_allels[tx_site_name]
                windows = []
                new_sites_name = []
                for site_allele in list_sites_of_allele:
                    windows.append(site_allele[3])
                    new_sites_name.append(site_allele[0])
                for i, row in tx_site_df.iterrows():
                    scores = []
                    seq = row['alignment_w_del']
                    for SNP_window in windows:
                        _, curr_score = get_snp_locus_and_score(seq, SNP_window)
                        scores.append(curr_score)
                    tx_site_df.loc[i,'allele'] = new_sites_name[np.argmin(scores)]
                for new_site in new_sites_name:
                    tx_site_df_temp = tx_site_df.loc[tx_site_df['allele'] == new_site]
                    tx_site_df_temp = tx_site_df_temp.drop(labels='allele', axis=1)
                    self._new_tx_df[new_site] = tx_site_df_temp

        return self._new_tx_df




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
    for i in range(len(curr_amplicon) - 20):
        try:
            window = curr_amplicon[i: i + 21]
        except:
            break
        curr_gap_score = count_diff(SNP_window, window)
        if curr_gap_score < smallest_gap:
            smallest_gap = curr_gap_score
            i_smallest = i

    return i_smallest + 10, smallest_gap





















