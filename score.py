from pesq import pesq
from pystoi import stoi
from sys import argv
import numpy as np
import librosa
import time
import os


def get_stoi2(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """

    stoi_val = 0
    try:
        stoi_val = stoi(ref_sig, out_sig, sr, extended=False)
    except:
        print("stoi error")
    return stoi_val


def get_pesq2(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ

    """
    pesq_val = 0
    pesq_val = pesq(sr, ref_sig, out_sig, 'wb')

    return pesq_val

def add_log(line,file_name = "./log.txt"):
    with open(file_name, 'a') as f:
        f.write(line+"\n")

def evaluate2(ref_dir, out_dir, extension="_enhanced", filename="./stoi_pesq.csv", sr=16000):
    ref_files = os.listdir(ref_dir)
    out_files = [rf.replace(".wav","")+extension+".wav" for rf in ref_files]
    print(len(out_files))
    times_stoi = 0
    times_pesq = 0
    pesq = 0
    stoi = 0
    nb_total = 0

    offset=0
    add_log("filename"+";"+"PESQ"+";"+"STOI", file_name=filename)
    for i,rf,of in zip(np.arange(len(ref_files)-offset),ref_files[offset:], out_files[offset:]):

        if i%100==0:
            print(i+offset,"/",len(ref_files))
        #print(i+offset,"/",len(ref_files),rf)
        ref_sig, _ = librosa.load(ref_dir+rf, sr=sr)
        out_sig, _ = librosa.load(out_dir+of, sr=sr)

        t1_stoi = time.time()
        stoi_i = get_stoi2(ref_sig, out_sig, sr)
        t2_stoi = time.time()
        if stoi_i < 1e-03:
            print(i+offset,"/",len(ref_files),rf,"stoi=",stoi_i)
        times_stoi += t2_stoi-t1_stoi
        stoi += stoi_i

        try:
            t1_pesq = time.time()
            pesq_i = get_pesq2(ref_sig, out_sig, sr)
            t2_pesq = time.time()
            times_pesq += t2_pesq-t1_pesq
            if np.isnan(pesq_i):
                print(i+offset,"/",len(ref_files),rf, "nan")
                add_log(filename+": "+rf)
            else:
                pesq+=pesq_i
                add_log(rf+";"+str(pesq_i)+";"+str(stoi_i), file_name=filename)
                nb_total+=1
        except:
            print(i+offset,"/",len(ref_files),rf, "pesq error !")
            add_log(filename+": "+rf)


        #t1_pesq = time.time()
        #pesq = get_pesq(ref_sig, out_sig, sr)
        #t2_pesq = time.time()
        #print("pesq",pesq, (t2_pesq-t1_pesq),"s")


    print("nb",nb_total,"/",len(ref_files),"\nstoi",stoi/nb_total, "runtime",times_stoi,"s","\npesq",pesq/nb_total, "runtime",times_pesq,"s")



#evaluate2("D:/DL4S_STT/Debruitage/mini_datasetV2.01_test/clean/", "D:/DL4S_STT/Debruitage/mini_datasetV2.01_test/noisy/")
#evaluate2("D:/DL4S_STT/Debruitage/datasetV2.01_test_train/test/clean/", "D:/DL4S_STT/Debruitage/datasetV2.01_test_train/test/noisy/")
#evaluate2("D:/DL4S_STT/Debruitage/datasetV2.01_test_train/test/clean/", "D:/DL4S_STT/Debruitage/denoised/master64/")
#print(get_pesq2(ref_sig, out_sig, 16000))
#evaluate2("D:/DL4S_STT/Debruitage/datasetV1.01_test_train/test/clean/", "D:/DL4S_STT/Debruitage/denoised/datasetv1/master64/")
#evaluate2("D:/DL4S_STT/Debruitage/datasetV1.01_test_train/test/clean/", "D:/DL4S_STT/Debruitage/denoised/datasetv1/dns48/")
#evaluate2("D:/DL4S_STT/Debruitage/datasetV1.01_test_train/test/clean/", "D:/DL4S_STT/Debruitage/denoised/datasetv1/dns64/")


def getopts(argv): #from https://gist.github.com/dideler/2395703
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0:2] == '--':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

if __name__ == '__main__':
    myargs = getopts(argv)
    if '--ref_dir' in myargs and "--noisy_dir" in myargs:
        clean_dir = myargs['--ref_dir']
        noisy_dir = myargs['--noisy_dir']
        file_name = myargs['--filename']
        if "--name_extension" in myargs:
            name_extension = myargs['--name_extension']
        else:
            name_extension = ""

        t1_eval = time.time()
        evaluate2(clean_dir, noisy_dir, filename=file_name, extension=name_extension)
        t2_eval = time.time()
        print("total", t2_eval-t1_eval,"s")
    else:
        print("python score.py --ref_dir <path of ref/clear dir> --noisy_dir <path of dir to be scored> (--name_extension <str extension of noisy audio files (for example _enhanced)>)")
        print("example:\n\n python score.py --ref_dir ./demo_clean/ --noisy_dir ./demo_noised/ --name_extension _enhanced --filename stoi_pesq.csv")
