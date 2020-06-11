import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn.feature_selection as sk
from scipy.stats import pearsonr
import math
import pandas as pd

import copy


def main():
    """
    Aa
    """
#    embd_file = 'glove.6B.50d.txt'
#    embd_file = 'glove.6B.100d.txt'
    embd_file = 'glove.6B.200d.txt'
#    embd_file = 'glove.6B.300d.txt'
    rduc_dim = 100   # the dimensions to reduce to
    kld = []   # KLD between INT (with no PCA) and the original Float
    acc_int = []   # accuracy using INT
    acc_int_pca = []   # accuracy using INT and PCA to 'rduc_dim' dimensions
    acc_int_wod = []   # accuracy using INT w/o dominant components
    # accuracy using PCA on the the INT w/o dominant after reducing to
    # 'rduc_dim' dimensions
    acc_int_pca_wod = []
    acc_flt = []   # accuracy of the float which is the best case
    acc_int_aftr_pca = []   # accuracy of INT after PCA on float
    corl_int_pca = []
    corl_pca_int = []
    # accuracy using INT on PCA after removing dominant component
    acc_int_pca_wodom = []
    # sum of all correlations for PCA on INT
    cor_pca_on_int = []
    # sum of all correlations for INT on PCA
    cor_int_on_pca = []

    # arrays to collect the semantic and syntactic accuracies
    sem_acc_int = []
    syn_acc_int = []
    sem_acc_int_pca = []
    syn_acc_int_pca = []
    sem_acc_int_wod = []
    syn_acc_int_wod = []
    sem_acc_int_pca_wod = []
    syn_acc_int_pca_wod = []
    sem_acc_flt = []
    syn_acc_flt = []

    # main function
    std_x = []   # the STD multiplier factor
    std_mult = 3
#    n_bits = 8   # number of bits to use in each dimension
    # go over all STD multiplier factors
#    for std_mult in range(2, 10, 2):
    for n_bits in range(3, 11, 1):
#        print('\nrun with ' + str(std_mult) + ' multiplier for STD')
        print('\nrun with ' + str(n_bits) + ' bits')
#        std_x.append(std_mult)
        std_x.append(n_bits)
        # calculate results for this setting of bits and std multiplier
        reslt = gen_matrix(n_bits, std_mult, embd_file, rduc_dim)
        # result will be an array with 6 resuls
        # [kl_fint, ac, ac_pca, ac_wod, ac_wod_pca, ac_flt]
        # kl_fint - KLD between int and float,
        # ac - accuracy of INT
        # ac_pca - accuract of INT PCA,
        # ac_wod - accuracy of INT w/o dominant component,
        # ac_wod_pca - accuracy of INT w/o dominant component PCA
        # ac_flt  -accuracy of the float
        kld.append(reslt[0])
        acc_int.append(reslt[1][0])
        acc_int_pca.append(reslt[2][0])
        acc_int_wod.append(reslt[3][0])
        acc_int_pca_wod.append(reslt[4][0])
        acc_flt.append(reslt[5][0])
        acc_int_aftr_pca.append(reslt[6][0])
        corl_int_pca.append(reslt[7])
        corl_pca_int.append(reslt[8])
        acc_int_pca_wodom.append(reslt[9][0])
        cor_pca_on_int.append(reslt[10])
        cor_int_on_pca.append(reslt[11])

        sem_acc_int.append(reslt[1][1])
        syn_acc_int.append(reslt[1][2])
        sem_acc_int_pca.append(reslt[2][1])
        syn_acc_int_pca.append(reslt[2][2])
        sem_acc_int_wod.append(reslt[3][1])
        syn_acc_int_wod.append(reslt[3][2])
        sem_acc_int_pca_wod.append(reslt[4][1])
        syn_acc_int_pca_wod.append(reslt[4][2])
        sem_acc_flt.append(reslt[5][1])
        syn_acc_flt.append(reslt[5][2])

    # generate an array with all elements equal to the true float accuracy
    # which is meassured only for the first run
    bst_acc = [acc_flt[0] for x in acc_flt]
    bst_sem = [sem_acc_flt[0] for x in sem_acc_flt]
    bst_syn = [syn_acc_flt[0] for x in syn_acc_flt]

    # plots
    sns.set(color_codes=True)

    plt.figure()
    plt.plot(std_x, kld, 'r')
    plt.ylabel('KLD')
#    plt.xlabel('Normalization factor')
    plt.xlabel('Number of bits')

    plt.figure()
    plt.plot(std_x, acc_int, 'r', std_x, acc_int_wod, 'b',
             std_x, bst_acc, 'g--')
    plt.ylabel('Accuracy of INT (r) INT on DCR (b) [%]')
#    plt.xlabel('Normalization factor')
    plt.xlabel('Number of bits')

    plt.figure()
    plt.plot(std_x, acc_int_aftr_pca, 'r', std_x, acc_int_pca, 'b')
    plt.ylabel('Accuracy PI (r) IP (b) [%]')
#    plt.xlabel('Normalization factor')
    plt.xlabel('Number of bits')

    plt.figure()
    plt.plot(std_x, acc_int_pca_wod, 'r', std_x, acc_int_pca_wodom, 'b')
    plt.ylabel('Accuracy IDP (r) DPI (b) [%]')
#    plt.xlabel('Normalization factor')
    plt.xlabel('Number of bits')

#    plt.figure()
#    plt.plot(std_x, corl_int_pca, 'r', std_x, corl_pca_int, 'b')
#    plt.ylabel('Correlation INT->PCA (r) PCA->INT (b) [%]')
#    plt.xlabel('Normalization factor')

    plt.figure()
    plt.plot(std_x, cor_pca_on_int, 'r', std_x, cor_int_on_pca, 'b')
    plt.ylabel('Toatl correlation PCA->INT (r) INT->PCA (b)')
#    plt.xlabel('Normalization factor')
    plt.xlabel('Number of bits')


def gen_matrix(bits, std_x, file, n_cmp):

    file_path = '/Users/rafifried/Programming/Python/NLP/Keras/glove.6B/'
    file_name = file
#    file_name = 'glove.6B.100d.txt'
    vectors_file = file_path + file_name

    # int_max will be 127 for 8bit and 15 for 4bit casting
    int_max = pow(2, bits) - 1
    # std_fctr will be the factor by which we 'widen' the distribusion in an
    # attmpt to have the full scale, within +/-'int_max', with about +/-3STD
    std_fctr = std_x

    # initiate dict 'vectors' whos keys are the words and the value for each
    # key is its embedded word vector as a list with all values for all
    # dimensions in the vector
    vectors = {}
    # initiate a list 'words' that will have all words in the vocabulary
    words = []
    # open the vectors.txt file and reads its line
    with open(vectors_file, 'r') as f:
        for line in f:
            # 'vals' will be a list with all strings in the current line
            # the 1st string is the word and all the rest are the vecor
            # dimensions values
            vals = line.rstrip().split(' ')
            # generate the dict entry with key being the word in the current
            # line and the value for this key is a list with the vector values
            vectors[vals[0]] = list(map(float, vals[1:]))
            # add the word in this line to the list of all words
            words.append(vals[0])

    # we will generate 2 dicts. 'vocab' will have an index value for each
    # key word. 'ivocab' will be exactely the opposite and will have the
    # word as the value for each of the numerical indexes
    vocab_size = len(words)   # total number of all words
#    print('\ntotal number of words is : ' + str(vocab_size))
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    # vector_dim is the embedding vector dimension
    vector_dim = len(vectors[ivocab[0]])
#    print('vectors dimensions are : ' + str(vector_dim))
    # initiate a matrix 'W' with all values zero. The number of rows in the
    # matrix is the number of words we have and the number of columns (length
    # of each row) will be the embedding vector dimension
    W = np.zeros((vocab_size, vector_dim))
    # vectors.items() has all key:value pairs in the dict vectors, as
    # tuples. 'word' will be the key, which in 'vectors' are the words, and 'v'
    # will be the value for this key, which in 'vectors' is the list of all
    # dimensions values in the the embedding vector for this word
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        # Every row in the matrix W will get its full embedding vector. The row
        # number will be according to the value for the key 'word' in the
        # 'vocab' dict, which is the word's index. So 'W' will practically
        # convert the dict 'vectors' into a matrix
        W[vocab[word], :] = v
#    print('\nthe weight matrix before normalization looks like this:\n')
#    for i in range(3):
#        print(W[i])
    # normalize each word vector to unit variance
    # 'W_norm' is initiated to be a matrix with the same shape as 'W' and all
    # zeros
    W_norm = np.zeros(W.shape)
    # for each row = word the square value in each column will be sumed
    # along the whole row (this is the 1 in (W**2, 1) that defines summation
    # along rows).
    # The result will be a vector with 1 column and the number of rows as W.
    # Then a square root of each row will be taken such that 'd' will be
    # a vecor with the sqrt of the sum of all squared values in
    # each dimeension for each word in the vocabulery = the vector size of
    # each word. 'd' will be 1 column and the number of rows is the number of
    # words. In each row is the vecotr length of the word
    d = (np.sum(W ** 2, 1) ** (0.5))
    # 'W_norm' will have the values of the normalized embedding vectors for
    # each word. Each word will be divided by its vector size
    # ****
    # W_norm is the normalized float embedded vector array
    # ***
    W_norm = (W.T / d).T
#    print('\nthe weight matrix after normalization looks like this:\n')
#    for i in range(3):
#        print(W_norm[i])
    # Generate the weight matrix that has int values for the word vectors
    # int_max will be 127 for 8bit or 15 for 4bit. Clip the weights to not
    # go below -127 or above 127 and then cast them into 8bit INT (-128, 127).
    # The gaussian distribution of the weights prior to normalization into
    # INT seem to have low STD and hence we can 'widen' it by the factor
    # 'std_fctr' to improve discremination/seperration between words.
    # int_ will be a 32/64bit int and int8 is an 8bit Byte (-128 to +127)
    # ***
    # W_int is the casted INT embedded vector array
    # ***
    W_int = np.int32(np.clip(W_norm * std_fctr * int_max, -int_max, int_max))
#    print('\nthe weight matrix after conversion to int looks like this:\n')
#    for i in range(3):
#        print(W_int[i])

    # ***
    # Find the mean vector of W_norm = the normalaized float embedding array
    # and of W_int = the array of the cassted INT. and generate embedding
    # matces without the mean
    # ***

    # the mean vector has the mean value in each diemnsion
    v_mean = np.mean(W_norm, axis=0)
#    print('\nshape of mean vector : ' + str(v_mean.shape))
#    print('/nmean vector:')
#    print(v_mean)
#    v_size = np.linalg.norm(v_mean)
#    print('vector size = ' + str(v_size))
    # subtract the mean vector from each word
    W_norm_iso = np.subtract(W_norm, v_mean)
    # find the mean vector for the INT embedding matrix
    v_mean_int = np.mean(W_int, axis=0)
    # subtract the mean from each word vector
    W_int_iso = np.subtract(W_int, v_mean_int)

    # ***
    # PCA on the floating point embedding matrix W_norm
    # The transformed matrix to the dominant components is W_pca
    # ***

    # find the number of principal components to retain 90% of the variance
    # in the origianl database
#    pca = PCA(0.9)
    pca = PCA(n_components=n_cmp)
    # define the number of principal componenst to be 50
#    pca = PCA(n_components=10)
    pca.fit(W_norm)   # run the pca on the original database
    # print the number of principle componets
#    print('\nPCA components of float matrix = ' + str(pca.n_components_))
    # print the sizes/variances of each principle componet
#    print('PCA sizes:')
#    print(pca.explained_variance_)
    # print the % of variance of the full data by each principle component
    pca_r = pca.explained_variance_ratio_
#    print('PCA ratios:')
#    print(pca_r)
    # print the sum of all variance by all components
    pca_t = sum(pca_r)
    print('PCA coverage of float = ' + str(pca_t))
    # project the original data base on the principal components
    # W_pca will be the transfomed embedding vector matrix
    W_pca = pca.transform(W_norm)
#    print('shape of matrix after PCA : ' + str(W_pca.shape))
    # convert to INT after doing the PCA on the original matrix
    W_int_aftr_pca =\
        np.int32(np.clip(W_pca * std_fctr * int_max, -int_max, int_max))
    print('shape of matrix of INT after PCA : ' + str(W_int_aftr_pca.shape))

    # ***
    # PCA on the float array w/o mean
    # pca_womean is the transformed embedding matrix
    # ***

    # find the number of principal components to retain 90% of the variance
    # in the origianl database
    # define the percentage of variance to capture
    pca_womean = PCA()
    # define the number of principal componenst to be 50
#    pca = PCA(n_components=50)
    pca_womean.fit(W_norm_iso)   # run the pca on the original database
    # print the number of principle componets
#    print('\nPCA components w/o mean = ' + str(pca_womean.n_components_))
    pca_womean_r = pca_womean.explained_variance_ratio_
#    print('PCA ratios for w/o mean:')
#    print(pca_womean_r)
    # print the sum of all variance by all components
    pca_womean_t = sum(pca_womean_r)
#    print('PCA coverage for w/o mean = ' + str(pca_womean_t))

    # ***
    # Subtract the 1st dominant component from all word vector in the float
    # matrix w/o mean
    # ***

    # all the principal vectors
    u_p = pca_womean.components_
#    print('\nshape of u_p : ' + str(u_p.shape))
    # the main dominant component
    u1 = u_p[0]
#    print('shape of u1 is : ' + str(u1.shape))
    # find the size of the projection of each word vector on the main dominant
    # component. this will be a list of 400000 scalar values
    proj_size = np.dot(u1.T, W_norm_iso.T)
#    print('shape of proj_size is : ' + str(proj_size.shape))
    # for each vector, based on its scalar value of size along the dominant
    # component, generate its own dominant component vector by scaling
    # the same dominant component by the size of projection of this word vector
    v_proj = np.asarray([np.multiply(x, u1) for x in proj_size])
#    print('shape of v_proj is : ' + str(v_proj.shape))
    # subtract the projection of each word vector along the principal component
    # from the word vector
    W_iso = W_norm_iso - v_proj

    # ***
    # PCA on the float array w/o the dominant component
    # ***

    # find the number of principal components to retain 90% of the variance
    # in the origianl database
    pca_wodom = PCA(n_components=n_cmp)
    # define the number of principal componenst to be 50
#    pca = PCA(n_components=50)
    pca_wodom.fit(W_iso)   # run the pca on the original database
    # print the number of principle componets
#    print('\nPCA components w/o dominant = ' + str(pca_wodom.n_components_))
    pca_wodom_r = pca_wodom.explained_variance_ratio_
#    print('PCA ratios for w/o dominant:')
#    print(pca_wodom_r)
    # print the sum of all variance by all components
    pca_wodom_t = sum(pca_wodom_r)
    print('PCA coverage for w/o dominant = ' + str(pca_wodom_t))
    # project W_iso on the principal components
    W_pca_wodom = pca_wodom.transform(W_iso)
    # convert to INT after doing the PCA on the matrix w/o dominant component
    W_int_pca_wodom =\
        np.int32(np.clip(W_pca_wodom * std_fctr * int_max, -int_max, int_max))



    # ***
    # PCA on the INT embedding matrix W_int
    # The transformed matrix to the dominant components is W_pca_int
    # ***

    # find the number of principal components to retain 90% of the variance
    # in the origianl database
#    pca_int = PCA()
    # define the number of principal componenst to be 50
    pca_int = PCA(n_components=n_cmp)
    pca_int.fit(W_int)   # run the pca on the original database
    # print the number of principle componets
#    print('\nPCA components for the INT = ' + str(pca_int.n_components_))
    # print the sizes/variances of each principle componet in INT
#    print('PCA sizes for INT:')
#    print(pca_int.explained_variance_)
    # print the % of variance of the full data by each principle component
    pca_int_r = pca_int.explained_variance_ratio_
#    print('PCA ratios for INT:')
#    print(pca_int_r)
    # print the sum of all variance by all components
    pca_int_t = sum(pca_int_r)
    print('PCA coverage for INT = ' + str(pca_int_t))

    # project W_int on the principal components
    W_pca_int = pca_int.transform(W_int)
#    print('shape of matrix after PCA for INT : ' + str(W_pca_int.shape))

    # ***
    # PCA on the INT array w/o the mean
    # ***

    pca_int_womean = PCA()
    pca_int_womean.fit(W_int_iso)   # run the pca on the original database
#    print('\nPCA components w/o mean of INT = '
#          + str(pca_int_womean.n_components_))
    pca_int_womean_r = pca_int_womean.explained_variance_ratio_
#    print('PCA ratios for w/o mean:')
#    print(pca_int_womean_r)

    # ***
    # Subtract the 1st dominant component for all the word vectors in the INT
    # matrix without mean
    # ***

    # all the principal vectors
    u_p_int = pca_int_womean.components_
#    print('\nshape of u_p_int : ' + str(u_p_int.shape))
    # the main dominant component
    u1_int = u_p_int[0]
#    print('shape of u1_int is : ' + str(u1_int.shape))
    # find the size of the projection of each word vector on the main dominant
    # component. this will be a list of 400000 scalar values
    proj_size_int = np.dot(u1_int.T, W_int_iso.T)
#    print('shape of proj_size_int is : ' + str(proj_size_int.shape))
    # for each vector, based on its scalar value of size along the dominant
    # component, generate its own dominant component vector by scaling
    # the same dominant component by the size of projection of this word vector
    v_proj_int = np.asarray([np.multiply(x, u1_int) for x in proj_size_int])
#    print('shape of v_proj_int is : ' + str(v_proj_int.shape))
    # subtract the projection of each word vector along the principal component
    # from the word vector
    W_iso_int = W_int_iso - v_proj_int

    # ***
    # PCA on the INT matrix without the 1st dominant component
    # ***

    # find the number of principal components to retain 90% of the variance
    # in the origianl database
#    pca_wodom_int = PCA()
    # define the number of principal componenst to be 50
    pca_wodom_int = PCA(n_components=n_cmp)
    pca_wodom_int.fit(W_iso_int)   # run the pca on the original database
    # print the number of principle componets
#    print('\nPCA components INT w/o dominant = ' +
#          str(pca_wodom_int.n_components_))
    pca_wodom_int_r = pca_wodom_int.explained_variance_ratio_
#    print('PCA ratios for INT w/o dominant:')
#    print(pca_wodom_int_r)
    pca_wodom_int_t = sum(pca_wodom_int_r)
    print('PCA coverage for INT w/o dominant = ' + str(pca_wodom_int_t))

    # project the W_iso_int on the proncipal components to get a matrix with
    # smaller dimension (in this case with only 90 dimensions)
    W_final = pca_wodom_int.transform(W_iso_int)
    print('shape of final matrix : ' + str(W_final.shape))


    # generate the transpose matrixes for the weights such that every embedding
    # dimension now is a row and every column now is a word. This is too make
    # it easy to visualize the distribution along each doemnsion
    vectors_t = W.T   # the transoped matrix prior to normalization
    vectors_nt = W_norm.T   # the transposed normalized matirx
    vectors_intt = W_int.T   # the transposed matrix after normalizing to INT
    # the transposed matrix after PCA and then normalizing to INT
    vectors_pca_int = W_int_aftr_pca.T
    # the transposed matrix after normalizing to INT and then PCA
    vectors_int_pca = W_pca_int.T
#    print('\nnumber of rows in the transposed array = ' + str(len(vectors_t)))
#    print('\nThe length of the transposed row is : ' + str(len(vectors_t[0])))

    # ***
    # check correlation between dimensions
    # ***
#    v_uniq = np.unique(vectors_nt[0])
#    print('\nnumber of unique float values = ' + str(len(v_uniq)))
#    v__int_uniq = np.unique(vectors_intt[0])
#    print('number of unique INT values = ' + str(len(v__int_uniq)))
    corr, _ = pearsonr(vectors_nt[10], vectors_nt[20])
#    print('\ncorrelation of float = ' + str(corr))
    corr_int, _ = pearsonr(vectors_intt[10], vectors_intt[20])
#    print('correlation of INT = ' + str(corr_int))
    corr_int_pca, _ = pearsonr(vectors_int_pca[10], vectors_int_pca[20])
    corr_pca_int, _ = pearsonr(vectors_pca_int[10], vectors_pca_int[20])

    # PCA after INT conversion to pandas
    e_pca_on_int = pd.DataFrame(W_pca_int)
    # INT after PCA conversion to pandas
    e_int_on_pca = pd.DataFrame(W_int_aftr_pca)
    # calculate correlation between all columns
    cor_pca_on_int = e_pca_on_int.corr(method='pearson')
    cor_int_on_pca = e_int_on_pca.corr(method='pearson')
    # sum all individual correlation between each column and all other columns
    tot_c_p_i = 0
    tot_c_i_p = 0
    # go over all columns
    for k in range(len(cor_pca_on_int)):
        # get the sum of the absolute value of the correlation of this column
        # k with all other column and subtract the self correlation which is 1.
        # we use the abs() since we are interested only in the 'magnitude' of
        # the correlations and not their 'direction'
        c_sum_p_on_i = sum(list(map(abs, cor_pca_on_int[k]))) - 1
        c_sum_i_on_p = sum(list(map(abs, cor_int_on_pca[k]))) - 1
        # add to the total correlation of all columns
        tot_c_p_i += c_sum_p_on_i
        tot_c_i_p += c_sum_i_on_p
    print('corr of pca on int = ' + str(tot_c_p_i))
    print('corr of int on pca = ' + str(tot_c_i_p))


    # generate a random distribution to represent the vocubelary
#    p1 = np.random.normal(0, 0.1, vocab_size)
#    s = pd.Series(p1)
#    p = (s.groupby(s).transform('count') / len(s)).values
#    p = p / np.sum(p)
#    print('sum of p = ' + str(np.sum(p)))

    # ***
    # calculate sum of KLD of all dimensions
    # ***
    total_kld = 0
    for j in range(len(vectors_nt)):
        q1 = vectors_nt[j]   # first dimaneion for float
        s = pd.Series(q1)
        q = (s.groupby(s).transform('count') / len(s)).values
        q = q / np.sum(q)
#        print('sum of q = ' + str(np.sum(q)))

        q1_int = vectors_intt[j]   # first dimension for INT
        s = pd.Series(q1_int)
        q_int = (s.groupby(s).transform('count') / len(s)).values
        q_int = q_int / np.sum(q_int)
#        print('sum of q_int = ' + str(np.sum(q_int)))

        # KLD for using the float dimension 0 to represent a perfect gaussian
#        kl_f = np.sum(np.where(p != 0, p * np.log2(p / q), 0))
        # KLD for using the INT dimension 0 to represent a perfect gaussian
#    kl_int = np.sum(np.where(p != 0, p * np.log2(p / q_int), 0))
        # KLD for using the INT dimension 0 to represent the float dimension 0
        kl_fint = np.sum(np.where(q != 0, q * np.log2(q / q_int), 0))
#        print('kl_f = ' + str(kl_f))
#        print('kl_int = ' + str(kl_int))
#        print('kl_fint = ' + str(kl_fint))
        total_kld += kl_fint

    # ***
    # plots
    # ***

    sns.set(color_codes=True)

    """
    # scatter plot of the 1st and 2nd dimensions of the embedding vectors
    # over all words in the vocablary
    plt.scatter(W_norm[:, 10], W_norm[:, 20])
    plt.xlabel('v_10')
    plt.ylabel('v_20')

    # scatter plot of the 1st and 2nd dimensions of the INT embedding vectors
    # over all words in the vocablary
    plt.figure()
    plt.scatter(W_int[:, 0], W_int[:, 1])

    # distribution of the values of all words in dimension 0 out of the 100
    # after normalization
    plt.figure()
    plt.xlim(-1, 1)
    sns.distplot(vectors_nt[2], color='r', hist=False)
    sns.distplot(vectors_nt[5], color='g', hist=False)
    sns.distplot(vectors_nt[8], color='b', hist=False)
    plt.xlabel('vector value v_2(red) v_5(breen) v_8(blue)')
    plt.ylabel('PDF')
    """

    # distribution of the values of all words in dimension 0 out of the 100
    # after these values were casted into an INT in the int_max range
    plt.figure()
    plt.xlim(2 * (-int_max), 2 * int_max)
    sns.distplot(vectors_intt[0])
    plt.ylabel('PSD')
    plt.xlabel('Vecror values')

    """
    # calculate means and stds
    f_std = [np.std(x) for x in vectors_nt]
    f_mean = [np.mean(x) for x in vectors_nt]
    i_std = np.std(vectors_intt[0])
    i_mean = np.mean(vectors_intt[0])
#    print('\nmean and std for normalized float : ')
#    print(f_mean)
#    print(f_std)

    # the mean and std values of all dimensions
    plt.figure()
    plt.plot(f_mean, 'bo', f_std, 'ro')
    plt.ylabel('mean-blue std-red')
    plt.xlabel('vector dimension')

    # the variance ratio of the 100 principal components of the float matrix
    plt.figure()
    plt.plot(pca_r)
    plt.ylabel('variance ratio')
    plt.xlabel('vector dimension')

    # the variance ratio of the 100 principal components of the float matrix
    plt.figure()
    plt.plot(pca_r[:10], 'bo')
    plt.ylabel('variance ratio')
    plt.xlabel('vector dimension')
    """

    """
    # distribution of the values of all words in the dimensions with largest
    # and lowest mean values
    plt.figure()
    sns.distplot(vectors_nt[np.argmax(f_std)])
    sns.distplot(vectors_nt[np.argmin(f_std)])
    """

    """
    s = pd.Series(vectors_nt[np.argmax(f_std)])
    q_mx = (s.groupby(s).transform('count') / len(s)).values
    q_mx = q_mx / np.sum(q_mx)
    kl_qmx = np.sum(np.where(p != 0, p * np.log2(p / q_mx), 0))
    print('kl_qmx = ' + str(kl_qmx))
    s = pd.Series(vectors_nt[np.argmin(f_std)])
    q_mn = (s.groupby(s).transform('count') / len(s)).values
    q_mn = q_mn / np.sum(q_mn)
    kl_qmn = np.sum(np.where(p != 0, p * np.log2(p / q_mn), 0))
    print('kl_qmn = ' + str(kl_qmn))
    """
#    print('sum = ' + str(np.sum(vectors_nt[0])))
#    print('mean and std for int : ')
#    print(i_mean, i_std)


    # ***
    # evaluations
    # ***

    ac_flt = [100, 100, 100]   # default accuracy
    # run the float evaluation only once for each embedded file
#    if std_fctr == 1:
    if bits == 3:
        print('\nevaluating with float:')
#        print('*********************************************************\n')
        flag = 'f'
        ac_flt = evaluate_vectors(W_norm, vocab, ivocab, flag)

    """
    print('\nevaluating with float PCA:')
    print('*********************************************************\n')
    flag = 'f'
    evaluate_vectors(W_pca, vocab, ivocab, flag)

    print('\nevaluating with float w/o mean:')
    print('*********************************************************\n')
    flag = 'f'
    evaluate_vectors(W_norm_iso, vocab, ivocab, flag)

    print('\nevaluating float without dominant component:')
    print('*********************************************************\n')
    flag = 'f'
    evaluate_vectors(W_iso, vocab, ivocab, flag)
    """

    print('evaluating with INT:')
#    print('*********************************************************\n')
    flag = 'i'
    ac = evaluate_vectors(W_int, vocab, ivocab, flag)

    print('evaluating with PCA after INT:')
#    print('*********************************************************\n')
    flag = 'i'
    ac_pca = evaluate_vectors(W_pca_int, vocab, ivocab, flag)

    print('evaluating with INT w/o dominant component:')
#    print('*********************************************************\n')
    flag = 'i'
    ac_wod = evaluate_vectors(W_iso_int, vocab, ivocab, flag)

    print('evaluating with PCA of INT w/o dominant component:')
#    print('*********************************************************\n')
    flag = 'i'
    ac_wod_pca = evaluate_vectors(W_final, vocab, ivocab, flag)

    print('evaluating INT after PCA of float:')
#    print('*********************************************************\n')
    flag = 'i'
    ac_int_aftr_pca = evaluate_vectors(W_int_aftr_pca, vocab, ivocab, flag)

    print('evaluating INT after PCA on float w/o dominant component:')
#    print('*********************************************************\n')
    flag = 'i'
    ac_int_pca_wodom = evaluate_vectors(W_int_pca_wodom, vocab, ivocab, flag)


#    return([kl_fint, ac[0], ac_pca[0], ac_wod[0], ac_wod_pca[0], ac_flt[0]])
    return([total_kld, ac, ac_pca, ac_wod, ac_wod_pca, ac_flt, ac_int_aftr_pca,
            corr_int_pca, corr_pca_int, ac_int_pca_wodom,
            tot_c_p_i, tot_c_i_p])


def evaluate_vectors(W, vocab, ivocab, flag):
    """
    Evaluate the trained word vectors on a variety of tasks
    """

    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
        ]

#    filenames = ['capital-common-countries.txt']
    prefix = '/Users/rafifried/Programming/Python/NLP/Glove_test/question-data/'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0   # count correct semantic questions
    correct_syn = 0   # count correct syntactic questions
    correct_tot = 0   # count correct questions
    count_sem = 0   # count all semantic questions
    count_syn = 0   # count all syntactic questions
    count_tot = 0   # count all questions
    full_count = 0   # count all questions, including those with unknown words

    # go over all test files
    file_cnt = 0
#    for i in range(1):
    for i in range(len(filenames)):
        file_cnt += 1
        # open the file prefix/filenames[i]
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            # full_data will be a list of lists with all words in all
            # file lines
            full_data = [line.rstrip().split(' ') for line in f]
            # 'full count' will be the total accumulated count of all lines in
            # all files
            full_count += len(full_data)
            # 'data' will be a list of lists, similar to 'full_data' where each
            # list includes all the words in the line. Every word is a seperate
            # string. For example ['athens', 'greece', 'baghdad', 'iraq']
            data = [x for x in full_data if all(word in vocab for word in x)]
            """
            print('\nfull conut = ' + str(full_count))
            print('\ndata:\n')
            print(data)
            """
        """
        # check if any word in data is not in the 400K vocabulary
        all_words = list(vocab.keys())
        for l in data:
            for w in l:
                if w not in all_words:
                    print(str(w) + ' is missing')
        """
        # 'row' is going over all lists in 'data' which are the lines in the
        # current test file. Then we go over all the individual 'word' in each
        # row, and we use the vocab[word] which is the index of the word, to
        # populate a list which will go into the array 'indices', such that it
        # will include lists of all indexes of all words in each line in the
        # current file.
        # 'indices' will be a 2D array with a row for every line in the current
        # file and in every row there will be 4 indices per the 4 words in the
        # line
        indices = np.array([[vocab[word] for word in row] for row in data])
        # since every line in all files, has 4 words and hence 4 indices, ind1
        # will be a list of indeces for all words in the 1st location in all
        # the lines in the file, ind2 the words in 2nd location, ind3 the 3rd
        # and ind4 the 4th
        ind1, ind2, ind3, ind4 = indices.T
        """
        print('\nindices:\n')
        print(indices[:10])
        print('\nind1:\n')
        print(ind1)
        print('\nind2:\n')
        print(ind2)
        """

        # 'predictions' is initiated as a vector of zeros with an entry
        # (0.0 float) for every line in the current file
        predictions = np.zeros((len(indices),))
        # num_iter will be the rounded up int value of the number of lines in
        # the current file divided by split_size which is initiated to 100,
        # such that it will be the number of split_size=100 chunks of lines in
        # indices
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        """
        print('\nnum_iter = ' + str(num_iter))
        """
        # go over all 100 line chunks
        for j in range(num_iter):
            # generate an array with index numbers in groups of split_size
            # which is 100. It will be 0 to 99 for the 1st chunk, then 100 to
            # 199 in the 2nd chunk and so on ...
            subset = np.arange(j*split_size, min((j + 1) *
                                                 split_size, len(ind1)))
            """
            print('\nsubset\n')
            print(subset)
            """

            # take the vectors/rows in W for the words whose indeces are in
            # ind2 (these are all the 2nd words in the lines whose indices are
            # in subset = in this chunk) subtruct from them the vectors/rows
            # in W for the words whose indeces are in ind1 (these are the 1st
            # words in the lines whose indices are in subset = this chunk),
            # and add to them the vectors/rows in W for the words whose
            # indeces are in ind3 (these are the 3rd words in the lines whose
            # indices are in subset = this chunk).
            # So practically we are getting all the 100 (length of subset)
            # (word2 - word1 + word3) vectors for the lines in the file whose
            # indices are in subset = this chunk. These vectors should be as
            # close as possible to word4.
            # pred_vec will have number of rows as the number of indices in the
            # subset, which is 100, and every row will have a vector in the
            # length of the words embedding vectors. This vector should be
            # close to the embedded vector of word4

            # calcualte using the full precision
            if flag == 'f':
                pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                            + W[ind3[subset], :])
            # calculate using 1byte precision
            else:
                # clip the values of the (w2-w1+w3) vector in all its
                # dimensions to [-127,127] and cast it into an INT
                pred_vec = np.int32(np.clip((W[ind2[subset], :]
                                            - W[ind1[subset], :]
                                            + W[ind3[subset], :]), -127, 127))
#            print('\nmax value in pred_vac array : ' + str(np.amax(pred_vec)))   # **
#            print('min value in pred_vac array : ' + str(np.amin(pred_vec)))   # **

            # cosine similarity.
            # Multiply the W weight matrix with the tarnsposed pred_vec matrix
            # (which is expected to be the embedded vector of the 4th words).
            # Since a dot multiplication is related to the distance between
            # vectors then we should be able to see the closest words. These
            # will be the ones with the highest values. The cosine of the angle
            # theta between vectors is the dot product of the vectors divided
            # by their length. Here we don't divide by the vectors lengthes
            # W : all words x embedding vectors
            # pred_vec.T : embedding vectors x 100 (indices number in subset)
            # dist : all words x 100 (the lines in file whose indices are in
            # subset)
            # we expect in each of the 100 columns that the higest value will
            # be in the row that coresponds to the word which is the 4th word
            # in the line (of the 100 lines of the subset)
            dist = (np.dot(W, pred_vec.T))

#            print('\ndata type of dist is : ' + str(dist.dtype))   # ************
#            print('max value in dist array : ' + str(np.amax(dist)))   # ********
#            print('min value in dist array : ' + str(np.amin(dist)))   # ********

            """
            print('\ndist dimensions:\n')
            print('rows = ' + str(len(dist)))
            print('columns = ' + str(len(dist[0])))
            print(dist[:1])
            """
            # k will go from 0 to 99 assuming 100 entries in the subset (which
            # are just the indeces of 100 lines in the file)
            # set to minus infinity the entry in the dist natrix in the coulmn
            # number k (the k-th line of the file) and the row that corresponds
            # to the 1st, 2nd and 3rd words in this line in the file
            for k in range(len(subset)):
#                dist[ind1[subset[k]], k] = -np.Inf
#                dist[ind2[subset[k]], k] = -np.Inf
#                dist[ind3[subset[k]], k] = -np.Inf
                dist[ind1[subset[k]], k] = -100000
                dist[ind2[subset[k]], k] = -100000
                dist[ind3[subset[k]], k] = -100000
            """
            print('\nupdated dist:')
            print(dist[ind1[subset[0]]])
            """

            # check the nearest neighbors words
            # get a list from the 1st column in dist. This will be the
            # distances of all words in the vocabulery from the eaxpected
            # 4th word in the 1st line in this subset. The index of  the
            # location in dist_nn will also be the index of the words in ivoab
            dist_nn = list(dist.T[0])
#            print('\nmax value of dist.T[0] = ' + str(dist.max()))
#            print('\nmin value of dist.T[0] = ' + str(dist.min()))
            knn_iwords = []   # indexes of the nearest words
            knn_words = []   # nearest words
            # look for the 7 closest words
            for nn in range(0, 7):
                max_w = 0   # initiate the similarity
                # go over all indexes in dist_nn
                for j in range(len(dist_nn)):
                    # find the word that has the largest similarity
                    if dist_nn[j] > max_w:
                        max_w = dist_nn[j]   # higest similarity thus far
                        max_wi = j   # index of word with higest similarity
                dist_nn[max_wi] = 0   # remove this word from the search list
                knn_iwords.append(max_wi)   # add the word's index to list
            # get the words whose indexes were collected
            for w_i in knn_iwords:
                knn_words.append(ivocab[w_i])
#            print('\nlooking for : ' + str(ivocab[ind4[subset[0]]]))   # ********
#            print('\nclosest nearest words : ')   # *****************************
#            print(knn_words)   # ************************************************

            # predicted word index
            # find the index of the max value in each column (the search along
            # columns is coming from the 0 in the argmax(dist, 0)).
            # 'predictions' was initiated as a vector of all zeros and with a
            # length per the number of lines in the file
            # It will now get the index of the predicted word (as shown in
            # vocab) for each line in the file according to its index in subset
            predictions[subset] = np.argmax(dist, 0).flatten()
            """
            print('\nindices of the max value in each column:\n')
            print(predictions)
            """

        # val will be True for every predictions entry that is equal to the 4th
        # word in the line
        val = (ind4 == predictions)   # correct predictions
        """
        print('\nval:\n')
        print(val)
        """

        """
        for x in range(len(val)):
            if not val[x]:
                print('wrong prediction at: ' + str(x))
                print('the word : ' + str(ind4[x]))
                print('the prediction : ' + str(predictions[x]))
        """

        # total of instances/lines tested for this file
        count_tot = count_tot + len(ind1)
        # total of correct predictions
        correct_tot = correct_tot + sum(val)
        # if we still didn't go over 5th file then check for semantics
        if i < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        # if we are over the 5th file then check for syntactics
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)

#        print("%s:" % filenames[i])
#        print('ACCURACY TOP1: %.2f%% (%d/%d)' %
#              (np.mean(val) * 100, np.sum(val), len(val)))

#    print('Questions seen/total: %.2f%% (%d/%d)' %
#          (100 * count_tot / float(full_count), count_tot, full_count))
#    print('Semantic accuracy: %.2f%%  (%i/%i)' %
#          (100 * correct_sem / float(count_sem), correct_sem, count_sem))
#    if i > 5:
#        print('Syntactic accuracy: %.2f%%  (%i/%i)' %
#              (100 * correct_syn / float(count_syn), correct_syn, count_syn))
#    print('Total accuracy: %.2f%%  (%i/%i)' %
#          (100 * correct_tot / float(count_tot), correct_tot, count_tot))
    tot_acrc = 100 * correct_tot / float(count_tot)
    sem_acrc = 100 * correct_sem / float(count_sem)
    if file_cnt > 6:
        syn_acrc = 100 * correct_syn / float(count_syn)
    else:
        syn_acrc = 0

    print('accuracies : ' + str([tot_acrc, sem_acrc, syn_acrc]))
#    print('total tests = ' + str(count_tot))
#    print('semantic tests = ' + str(count_sem))
#    print('syntactic tests = ' + str(count_syn))

    return([tot_acrc, sem_acrc, syn_acrc])


if __name__ == "__main__":
    main()
