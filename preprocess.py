import argparse
import os
import pandas as pd
import pickle as pkl
import healpy as hp
import numpy as np
import fitsio
from astropy.table import Table
from functools import reduce
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='data/original', \
                    help="Directory with the original dataset")
parser.add_argument('--output_dir', default='data/preprocessed_binary', \
                    help="Where to write the new dataset")


def cut(df):
    """Cut the detected galaxies out and shuffle the rest

    Args:
        df: Pandas DataFrame of detected and undetected galaxies
    """

    df = df[df['detected']==0]
    df = df.sample(frac=1).reset_index(drop=True) #shuffle
    return df.iloc[:2915104]


def modify_columns_and_concat(df_detect, df_undetect, skymaps, skymapnames):
    """ Add and remove columns from the detected and undetected DataFrames
    and concatenate them together

    Args:
        df_detect: Pandas DatFrame of detected galaxies
        df_undetect: Pandas DataFrame of undetected galaxies
        skymaps: maps containing the observing conditions
        skymapnames: names of the maps containing the observing conditions
    """

    df_undetect['pixel'] = hp.ang2pix(nside=4096, theta=df_undetect['true_dec'].apply(lambda dec: \
                                      np.deg2rad(90 - dec)), \
                                      phi = df_undetect['true_ra'].apply(lambda ra: np.deg2rad(ra)))
    df_undetect = df_undetect[['pixel','detected','true_g','true_r','true_i','true_z']]
    df_undetect[['true_g','true_r','true_i','true_z']] = df_undetect[['true_g','true_r','true_i',\
                                                                      'true_z']].apply(lambda m: \
                                                                       10**((30-m)/2.5))

    df_detect['pixel'] = hp.ang2pix(nside=4096, theta=df_detect['unsheared/dec'].apply(lambda dec: \
                                    np.deg2rad(90 - dec)), \
                                    phi = df_detect['unsheared/ra'].apply(lambda ra: np.deg2rad(ra)))
    df_detect['detected'] = np.ones(len(df_detect))
    df_detect[['true_g','true_r','true_i','true_z']] = df_detect[['BDF_FLUX_DERED_G','BDF_FLUX_DERED_R',\
                                                                  'BDF_FLUX_DERED_I','BDF_FLUX_DERED_Z']]
    df_detect = df_detect[['pixel','detected','true_g','true_r','true_i','true_z']]

    df = pd.concat([df_undetect, df_detect], ignore_index=True)
    
    select = np.array(df['pixel'])
    for (skymapname, skymap) in zip(skymapnames, skymaps):
        df[skymapname] = skymap[select]
    return df


def split(df):
    """Split the Pandas DataFrame into a Training
    test, and validation sets.

    Args:
        df: the Pandas DataFrame with all the data
    """

    df = df.sample(frac=1).reset_index(drop=True) #shuffle
    split = 10000
    test_df = df.iloc[:split]
    val_df = df.iloc[split:2*split]
    train_df = df.iloc[2*split:]
    return test_df, train_df, val_df


def split_and_normalize(df, output_dir):
    """Split the DataFrame and Normalize the results by the
    mean and standard deviation of the training set.

    Args:
        df: the Pandas DataFrame with all the data
        output_dir: the directory to save the mean and standard
            deviation of the training set to
    """

    test_df, train_df, val_df = split(df)
    train_mean = np.array(train_df.mean()[2:])
    train_std = np.array(train_df.std()[2:])
    train_df[[str(col) for col in train_df.columns][2:]] = (train_df[[str(col) for col in \
                                                            train_df.columns][2:]] - train_mean) / train_std
    test_df[[str(col) for col in test_df.columns][2:]] = (test_df[[str(col) for col in \
                                                          test_df.columns][2:]] - train_mean) / train_std
    val_df[[str(col) for col in val_df.columns][2:]] = (val_df[[str(col) for col in \
                                                        val_df.columns][2:]] - train_mean) / train_std
    pd.DataFrame({'train_mean': train_mean, 'train_std': train_std}, index=[str(col) for col in \
                    train_df.columns][2:]).to_pickle(os.path.join(output_dir,'normalization.pkl'))
    return {'test': test_df, 'train': train_df, 'val': val_df}


def preprocess(args):
    """Prepare the data to be fed into the neural network"""

    print("Loading detection catalog...")
    detectionfile = os.path.join(args.input_dir,'balrog_detection_catalog_sof_run2_v1.3.fits')
    detectioncols = ['true_bdf_mag_deredden','true_ra','true_dec','detected']
    detectiontable = Table(fitsio.read(detectionfile, columns=detectioncols).byteswap().newbyteorder())
    
    df_undetect = pd.DataFrame()
    df_undetect['true_g'] = detectiontable['true_bdf_mag_deredden'][:,0]
    df_undetect['true_r'] = detectiontable['true_bdf_mag_deredden'][:,1]
    df_undetect['true_i'] = detectiontable['true_bdf_mag_deredden'][:,2]
    df_undetect['true_z'] = detectiontable['true_bdf_mag_deredden'][:,3]
    for col in detectioncols[1:]:
        df_undetect[col] = detectiontable[col]
    
    print('Length of detection catalog: {}'.format(len(df_undetect))) #19959472
    print('Number of detected galaxies: {}'.format(len(df_undetect[df_undetect['detected'] == 1]))) #7364568
    print('Number of undetected galaxies: {}'.format(len(df_undetect[df_undetect['detected'] == 0]))) #12594904
    
    df_undetect = cut(df_undetect)
    
    print("Loading Balrog catalog...")
    df_detect = pd.read_pickle(os.path.join(args.input_dir, 'deep_balrog.pkl'))
    
    print("Reading sky maps...")
    conditions_dir = os.path.join(args.input_dir, 'conditions')
    skymapnames = os.listdir(conditions_dir)
    skymaps = [hp.read_map(os.path.join(conditions_dir,skymapname)) for skymapname in skymapnames]
    
    print("Combining and Modifying DataFrame columns...")
    df = modify_columns_and_concat(df_detect, df_undetect, skymaps, skymapnames)
    print("Normalizing the dataset...")
    split_df = split_and_normalize(df, args.output_dir)
    
    print("Saving preprocessed dataset...")
    for dslice in ['test', 'train', 'val']:
        saveloc = os.path.join(args.output_dir,dslice)
        true = torch.from_numpy(split_df[dslice][['true_g','true_r','true_i',\
                                                  'true_z']].values).float().contiguous()
        cond = torch.from_numpy(split_df[dslice][skymapnames].values).float().contiguous()
        out = torch.from_numpy(split_df[dslice]['detected'].values).float().contiguous()
        torch.save(true, os.path.join(saveloc,'true.pth'))
        torch.save(cond, os.path.join(saveloc,'cond.pth'))
        torch.save(out, os.path.join(saveloc,'out.pth'))


if __name__ == '__main__':
    args = parser.parse_args()
    
    assert os.path.isdir(args.input_dir), "Couldn't find the dataset at {}".format(args.input_dir)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Output dir {} already exists".format(args.output_dir))

    preprocess(args)


