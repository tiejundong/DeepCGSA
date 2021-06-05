#DeepCGSA demo

import optparse
import os
import shutil
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import logging
logging.getLogger('tensorflow').disabled = True

import pandas as pd
from biopandas.pdb import PandasPdb
from sklearn.preprocessing import OneHotEncoder
import scipy
import scipy.spatial
from Bio.PDB import Model as biopy_Model
from Bio.PDB import NACCESS


#define CG and residue types
CGAA_types = ['CA', 'CACB', 'martini', 'P', '3SPN', 'AA'] # SASA of AA structure was calculated by NACCESS
CG_types = ['CA', 'CACB', 'martini', 'P', '3SPN']
prot_resi_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
martini_bead_type = ['TRP_BB', 'TRP_SC1', 'TRP_SC2', 'TRP_SC3', 'TRP_SC4', 'TYR_BB', 'TYR_SC1', 'TYR_SC2', 'TYR_SC3', 'PHE_BB', 'PHE_SC1', 'PHE_SC2', 'PHE_SC3', 'HIS_BB', 'HIS_SC1', 'HIS_SC2',
 'HIS_SC3', 'HIH_BB', 'HIH_SC1', 'HIH_SC2', 'HIH_SC3', 'ARG_BB', 'ARG_SC1', 'ARG_SC2', 'LYS_BB', 'LYS_SC1', 'LYS_SC2', 'CYS_BB', 'CYS_SC1', 'ASP_BB', 'ASP_SC1', 'GLU_BB', 'GLU_SC1',
 'ILE_BB', 'ILE_SC1', 'LEU_BB', 'LEU_SC1', 'MET_BB', 'MET_SC1', 'ASN_BB', 'ASN_SC1', 'PRO_BB', 'PRO_SC1', 'HYP_BB', 'HYP_SC1', 'GLN_BB', 'GLN_SC1', 'SER_BB', 'SER_SC1', 'THR_BB',
 'THR_SC1', 'VAL_BB', 'VAL_SC1', 'ALA_BB', 'GLY_BB']
rna_resi_types = ['A', 'C', 'G', 'U']
mc_atom = ['N', 'CA', 'C', 'O']                                    #main chain atom
atom_w = {'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'S': 32.065}   #atom weight
coor_cols = ['x_coord', 'y_coord', 'z_coord']


class DeepCGSA():
    def __init__(self, CG_type):
        '''
        basic paramters
        '''
        self.CG_type = CG_type
        
        self.last_channels_1d = 3
        self.block_kernel_size_1d = 15
        self.n_repeat_1d = [0, 1]
        self.n_block_1d = np.array(self.n_repeat_1d)+1
        self.n_hidden_1d = [8, 16]
        self.block_kernel_size_2d = 5
        self.n_repeat_2d = [1, 2]
        self.n_block_2d = np.array(self.n_repeat_2d)+1
        self.n_hidden_2d = [8, 16]
        
        #several small differences between models
        if self.CG_type in ['P', '3SPN']:
            self.block_kernel_size_1d = 5
        if self.CG_type=='P':
            self.n_repeat_2d = [0, 1]
        if CG_type=='3SPN':
            self.n_hidden_2d = [12, 16]
            
        self.n_block_1d = np.array(self.n_repeat_1d)+1
        self.n_block_2d = np.array(self.n_repeat_2d)+1
        self.act = tf.nn.leaky_relu
        
        if self.CG_type=='martini':
            self.input_bead_type = martini_bead_type
        elif self.CG_type in ['P', '3SPN']:
            self.input_bead_type = rna_resi_types
        else:
            self.input_bead_type = prot_resi_types
        
        self.sasa_percentcut = None #99 percent cut for normalize
        self.distance_percentcut = None
        self.model = None
        self.weight_path = './model_weight/{}/model_weight.tf'.format(self.CG_type)
     
    def build_1d_resblock(self, input_, in_channels, out_channels, name):
        '''
        build a 1d resblock
        '''
        conv_1 = Conv1D(out_channels, kernel_size=self.block_kernel_size_1d, strides=1, padding="same", dilation_rate=1, use_bias=False, data_format='channels_last', name=name+'_conv_1')
        bn_1 = BatchNormalization(name=name+'_bn_1')
        conv_2 = Conv1D(out_channels, kernel_size=self.block_kernel_size_1d, strides=1, padding="same", dilation_rate=1, use_bias=False, data_format='channels_last', name=name+'_conv_2')
        bn_2 = BatchNormalization(name=name+'_bn_2')
        shortcut_conv = Conv1D(out_channels, kernel_size=1, strides=1, use_bias=False, data_format='channels_last', name=name+'_conv_shortcut')
        shortcut_bn = BatchNormalization(name=name+'_bn_shortcut')
        act_1 = Activation(self.act)
        act_2 = Activation(self.act)
        
        x = conv_1(input_)
        x = bn_1(x)
        x = act_1(x)
        x = conv_2(x)
        x = bn_2(x)
        if in_channels == out_channels:
            shortcut_x = input_
        else:
            shortcut_x = shortcut_conv(input_)
            shortcut_x = shortcut_bn(shortcut_x)
        x = add([x, shortcut_x])
        output = act_2(x)
        return output
    
    def std_conv1d(self, input_, channels, kernel_size):
        '''
        a simple conv1d with BN and activation
        '''
        x = Conv1D(channels, kernel_size=kernel_size, strides=1, padding="same", dilation_rate=1, use_bias=False)(input_)
        x = BatchNormalization()(x)
        x = Activation(self.act)(x)
        return x
    
    def build_2d_resblock(self, input_, in_channels, out_channels, name):  #use atrous_conv2d?
        '''
        build a 2d resblock
        ''' 
        conv_1 = Conv2D(out_channels, kernel_size=(self.block_kernel_size_2d, self.block_kernel_size_2d), strides=1, padding="same", dilation_rate=1, use_bias=False, data_format='channels_last', name=name+'_conv_1')
        bn_1 = BatchNormalization(name=name+'_bn_1')
        conv_2 = Conv2D(out_channels, kernel_size=(self.block_kernel_size_2d, self.block_kernel_size_2d), strides=1, padding="same", dilation_rate=1, use_bias=False, data_format='channels_last', name=name+'_conv_2')
        bn_2 = BatchNormalization(name=name+'_bn_2')
        shortcut_conv = Conv2D(out_channels, kernel_size=1, strides=1, use_bias=False, data_format='channels_last', name=name+'_conv_shortcut')
        shortcut_bn = BatchNormalization(name=name+'_bn_shortcut')
        act_1 = Activation(self.act)
        act_2 = Activation(self.act)
        
        x = conv_1(input_)
        x = bn_1(x)
        x = act_1(x)
        x = conv_2(x)
        x = bn_2(x)
        if in_channels == out_channels:
            shortcut_x = input_
        else:
            shortcut_x = shortcut_conv(input_)
            shortcut_x = shortcut_bn(shortcut_x)
        x = add([x, shortcut_x])
        output = act_2(x)
        return output

    def std_conv2d(self, input_, channels, kernel_size):
        '''
        a simple conv2d with BN and activation
        '''
        x = Conv2D(channels, kernel_size=(kernel_size, kernel_size), strides=1, padding="same", dilation_rate=1, use_bias=False)(input_)
        x = BatchNormalization()(x)
        x = Activation(self.act)(x)
        return x
    
    def build_model(self, output_relu=True):
        '''
        build model
        '''
        #input for residue sequence or bead type sequence (only for martini)
        input_seqbeadtype = Input((None, len(self.input_bead_type)))
        x_1 = input_seqbeadtype
        
        #1d FCN resnet
        if self.CG_type in ['CA', 'CACB', 'martini']:
            x_1 = self.std_conv1d(x_1, 4, 20)
            x_1 = self.std_conv1d(x_1, 8, 15)
            in_channels = 8
        else:
            x_1 = self.std_conv1d(x_1, 4, 5)
            in_channels = 4
        for i_repeat in range(len(self.n_repeat_1d)):
            out_channels = self.n_hidden_1d[i_repeat]
            for i_block in range(self.n_block_1d[i_repeat]):
                x_1 = self.build_1d_resblock(x_1, in_channels, out_channels, name='1dresnet_'+str(i_repeat)+'_block_'+str(i_block))
                in_channels = out_channels
        x_1 = self.std_conv1d(x_1, self.last_channels_1d, 1)
        x_1 = tf.expand_dims(x_1, -2)
        seq_len = tf.shape(x_1)[1]
        beadtype_output = tf.tile(x_1, [1, 1, seq_len, 1])
        
        #transpose 
        beadtype_output_T = tf.transpose(beadtype_output, perm=[0, 2, 1, 3])
        
        #inut for distance matrix 
        if self.CG_type=='3SPN':
            input_seqdismap = Input((None, None, 6))
        elif self.CG_type=='CACB':
            input_seqdismap = Input((None, None, 3))
        else:
            input_seqdismap = Input((None, None))
        x_2 = input_seqdismap
        if self.CG_type in ['CA', 'P', 'martini']:
            dismap_output = tf.expand_dims(x_2, -1)
        else:
            dismap_output = x_2
        
        #concat all
        x = tf.concat([beadtype_output, dismap_output, beadtype_output_T], -1)
        
        #2d FCN resnet
        if self.CG_type=='CACB':
            in_channels = self.last_channels_1d*2+3
        elif self.CG_type=='3SPN':
            in_channels = self.last_channels_1d*2+6
        else:
            in_channels = self.last_channels_1d*2+1
        for i_repeat in range(len(self.n_repeat_2d)):
            out_channels = self.n_hidden_2d[i_repeat]
            for i_block in range(self.n_block_2d[i_repeat]):
                x = self.build_2d_resblock(x, in_channels, out_channels, name='2dresnet_'+str(i_repeat)+'_block_'+str(i_block))
                in_channels = out_channels
                
        #output
        if self.CG_type=='3SPN':
            x = Conv2D(3, kernel_size=(1, 1), strides=1, padding="same", dilation_rate=1, use_bias=False)(x)
            x = tf.math.reduce_sum(x, axis=-2)
        elif self.CG_type=='CACB':
            x = Conv2D(2, kernel_size=(1, 1), strides=1, padding="same", dilation_rate=1, use_bias=False)(x)
            x = tf.math.reduce_sum(x, axis=-2)
        else:
            x = Conv2D(1, kernel_size=(1, 1), strides=1, padding="same", dilation_rate=1, use_bias=False)(x)
            x = tf.squeeze(x, axis=-1)
            x = tf.math.reduce_sum(x, axis=-1)
        
        #last activation
        if output_relu==False:
            #print('output with custom function')
            x = K.switch(x>0, x, -tf.math.log1p(-x))
        elif output_relu==True:
            #print('output with relu')
            x = Activation('relu')(x)
        else:
            #print('output with linear')
            x = tf.keras.activations.linear(x)
        output_ = x
        
        self.model = Model(inputs=(input_seqbeadtype, input_seqdismap), outputs=output_)
        #tf.keras.utils.plot_model(self.model, to_file='./model.png', show_shapes=True)
        #self.model.summary()
        
    def initial_model(self):
        '''
        build model, load weights and params for normalize
        '''
        norm_param = np.load('./model_weight/{}/norm_param.npy'.format(self.CG_type))
        self.sasa_percentcut = norm_param[6]
        self.distance_percentcut = norm_param[7]
        self.build_model(output_relu=True)
        self.model.load_weights(self.weight_path)

def delmkdir(path):
    '''
    remove a path and create a new one
    '''
    isexist = os.path.exists(path)
    if isexist == True : 
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)

def check_input(options):
    '''
    check input
    '''
    if not os.path.exists(options.pdb_filename):
        raise AssertionError('can\'t find input pdb file:', options.pdb_filename)
    if options.CG_type not in CGAA_types:
        raise AssertionError('can\'t find input CG type \'{}\' in available types {}'.format(options.CG_type, CGAA_types))
    if options.create_CG!=False:
        if options.create_CG not in CG_types:
            raise AssertionError('can\'t find CG type \'{}\' in available types {}'.format(options.create_CG, CG_types))
    df_atom = PandasPdb().read_pdb(options.pdb_filename).df['ATOM']
    if len(df_atom['chain_id'].unique())>1:
        print('Input file contained multiple chains, first chain will be used')

def create_testCGfile(AA_filename, CG_type):
    '''
    create CG pdb file from AA pdb file
    '''
    CG_path = '{}_{}.pdb'.format(AA_filename.split('.')[-2], CG_type)
    shutil.copyfile(AA_filename, CG_path)
    biodf_data = PandasPdb().read_pdb(CG_path)
    df_atom = biodf_data.df['ATOM']
    df_atom = df_atom[df_atom['chain_id'] == df_atom['chain_id'].unique()[0]]
    
    if CG_type=='CA':
        df_CA = df_atom[df_atom['atom_name'] == 'CA'].copy()
        biodf_data.df['ATOM'] = df_CA
        biodf_data.to_pdb(CG_path, records=['ATOM'])
    
    if CG_type=='CACB':
        df_CA = df_atom[df_atom['atom_name'] == 'CA'].copy()
        def map_atom(x):
            if x in mc_atom:
                return 'main_chain'
            else:
                return 'side_chain'
        df_atom['atom_type'] = df_atom['atom_name'].apply(lambda x: map_atom(x))
        df_sc = df_CA[['residue_name', 'residue_number']].copy()
        def map_sc_coor(x, dim):
            resi_name = df_CA[df_CA['residue_number']==x]['residue_name'].tolist()[0]
            if resi_name=='GLY':
                cen_coor = df_CA[df_CA['residue_number']==x][dim].tolist()[0]
                return cen_coor
            else:
                df_temp = df_atom[(df_atom['residue_number']==x) & (df_atom['atom_type']=='side_chain')][[dim, 'element_symbol']]
                df_temp['atom_weight'] = df_temp['element_symbol'].apply(lambda element_symbol: atom_w[element_symbol])
                coor_weight = df_temp[[dim, 'atom_weight']].values
                sum_coorw = np.sum(coor_weight[:,0] * coor_weight[:,1])
                sum_w = np.sum(coor_weight[:,1])
                cen_coor = sum_coorw / sum_w
                return cen_coor
        for coor in coor_cols:
            df_sc[coor] = df_CA['residue_number'].apply(lambda x: map_sc_coor(x, coor))
        df_CA = df_CA.reset_index(drop=True)
        df_sc = df_sc.reset_index(drop=True)
        df_CACB = pd.concat([df_CA,df_CA]).reset_index(drop=True)
        df_CACB['atom_number'] = df_CACB.index + 1
        df_CACB['line_idx'] = df_CACB.index + 1
        for i in range(len(df_CA)):
            resi_idx = i+1
            df_CACB.loc[i*2, 'residue_number'] = resi_idx
            df_CACB.loc[i*2, 'residue_name'] = df_CA.loc[i, 'residue_name']
            df_CACB.loc[i*2, coor_cols] = df_CA.loc[i, coor_cols].values
            df_CACB.loc[i*2+1, 'residue_number'] = resi_idx
            df_CACB.loc[i*2+1, 'residue_name'] = df_CA.loc[i, 'residue_name']
            df_CACB.loc[i*2+1, 'atom_name'] = 'CB'
            df_CACB.loc[i*2+1, coor_cols] = df_sc.loc[i, coor_cols].values
        biodf_data.df['ATOM'] = df_CACB
        biodf_data.to_pdb(CG_path, records=['ATOM'])
        
    if CG_type=='martini':
        shutil.copyfile(CG_path, './temp.pdb')
        os_return = os.system('python martinize.py -f {} -x {} -ff martini22'.format('./temp.pdb', CG_path))
        if os_return!=0:
            raise AssertionError('martinize error')
        os.remove('./temp.pdb')
        
    if CG_type=='P':
        df_P = df_atom[df_atom['atom_name'] == 'P'].copy()
        biodf_data.df['ATOM'] = df_P
        biodf_data.to_pdb(CG_path, records=['ATOM'])
        
    if CG_type=='3SPN':
        df_3SPN = df_atom[(df_atom['atom_name']=='P') |\
                         (df_atom['atom_name']=='C4\'') |\
                          (((df_atom['residue_name']=='A') | (df_atom['residue_name']=='G')) & (df_atom['atom_name']=='N1') | 
                            ((df_atom['residue_name']=='C') | (df_atom['residue_name']=='U')) & (df_atom['atom_name']=='N3'))
                         ].copy()
        df_3SPN = df_3SPN.reset_index(drop=True)
        df_3SPN.loc[df_3SPN[(df_3SPN['atom_name']=='C4\'')].index, ['atom_name']] = 'S'
        df_3SPN.loc[df_3SPN[(((df_3SPN['residue_name']=='A') | (df_3SPN['residue_name']=='G')) & (df_3SPN['atom_name']=='N1')) | 
                (((df_3SPN['residue_name']=='C') | (df_3SPN['residue_name']=='U')) & (df_3SPN['atom_name']=='N3'))].index, ['atom_name']] = 'N'
        biodf_data.df['ATOM'] = df_3SPN
        biodf_data.to_pdb(CG_path, records=['ATOM'])
        
        
def create_input_array(pdb_filename, CG_type):
    '''
    read CG pdb file, create distance matrix and one-hot sequence array
    '''
    biodf_data = PandasPdb().read_pdb(pdb_filename)
    df_atom = biodf_data.df['ATOM']
    df_atom = df_atom[df_atom['chain_id'] == df_atom['chain_id'].unique()[0]]
    
    if CG_type=='CA':
        bead_encoder = OneHotEncoder()
        bead_encoder.fit(np.array(prot_resi_types).reshape(-1, 1))
        bead_onehot = bead_encoder.transform(df_atom['residue_name'].values.reshape(-1, 1)).toarray()
        dismap = scipy.spatial.distance.cdist(df_atom[coor_cols], df_atom[coor_cols], metric='euclidean')
        
    if CG_type=='CACB':
        bead_encoder = OneHotEncoder()
        bead_encoder.fit(np.array(prot_resi_types).reshape(-1, 1))
        df_CA = df_atom[df_atom['atom_name'] == 'CA']
        df_CB = df_atom[df_atom['atom_name'] == 'CB']
        bead_onehot = bead_encoder.transform(df_CA['residue_name'].values.reshape(-1, 1)).toarray()
        CACA_dismap = scipy.spatial.distance.cdist(df_CA[coor_cols], df_CA[coor_cols], metric='euclidean')
        CBCB_dismap = scipy.spatial.distance.cdist(df_CB[coor_cols], df_CB[coor_cols], metric='euclidean')
        CACB_dismap = scipy.spatial.distance.cdist(df_CA[coor_cols], df_CB[coor_cols], metric='euclidean')
        dismap = np.concatenate([np.expand_dims(CACA_dismap, axis=-1),np.expand_dims(CBCB_dismap, axis=-1),np.expand_dims(CACB_dismap, axis=-1)], axis=-1)
    
    if CG_type=='martini':
        bead_encoder = OneHotEncoder()
        bead_encoder.fit(np.array(martini_bead_type).reshape(-1, 1))
        df_atom['cgaa'] = df_atom['residue_name'] +'_'+ df_atom['atom_name']
        bead_onehot = bead_encoder.transform(df_atom['cgaa'].values.reshape(-1, 1)).toarray()
        dismap = scipy.spatial.distance.cdist(df_atom[coor_cols], df_atom[coor_cols], metric='euclidean')
        
    if CG_type=='P':
        bead_encoder = OneHotEncoder()
        bead_encoder.fit(np.array(rna_resi_types).reshape(-1, 1))
        bead_onehot = bead_encoder.transform(df_atom['residue_name'].values.reshape(-1, 1)).toarray()
        dismap = scipy.spatial.distance.cdist(df_atom[coor_cols], df_atom[coor_cols], metric='euclidean')
        
    if CG_type=='3SPN':
        bead_encoder = OneHotEncoder()
        bead_encoder.fit(np.array(rna_resi_types).reshape(-1, 1))
        df_P = df_atom[df_atom['atom_name']=='P']
        df_sugar = df_atom[df_atom['atom_name']=='S']
        df_base = df_atom[df_atom['atom_name']=='N']
        dismap_list = []
        dismap_list.append(scipy.spatial.distance.cdist(df_P[coor_cols], df_P[coor_cols], metric='euclidean'))
        dismap_list.append(scipy.spatial.distance.cdist(df_P[coor_cols], df_sugar[coor_cols], metric='euclidean'))
        dismap_list.append(scipy.spatial.distance.cdist(df_P[coor_cols], df_base[coor_cols], metric='euclidean'))
        dismap_list.append(scipy.spatial.distance.cdist(df_sugar[coor_cols], df_sugar[coor_cols], metric='euclidean'))
        dismap_list.append(scipy.spatial.distance.cdist(df_sugar[coor_cols], df_base[coor_cols], metric='euclidean'))
        dismap_list.append(scipy.spatial.distance.cdist(df_base[coor_cols], df_base[coor_cols], metric='euclidean'))
        dismap = np.concatenate([x.reshape(len(df_P), len(df_P), 1) for x in dismap_list], axis=-1)
        bead_onehot = bead_encoder.transform(df_P['residue_name'].values.reshape(-1, 1)).toarray()
    
    return bead_onehot, dismap

def pred_sasa(model, bead_onehot, dismap, pdb_filename, CG_type):
    '''
    use DeepCGSA to give a prediction
    '''
    bead_onehot, dismap = np.expand_dims(bead_onehot, axis=0), np.expand_dims(dismap/model.distance_percentcut, axis=0)
    pred = model.model.predict([bead_onehot, dismap])
    pred = np.squeeze(pred, axis=0)*model.sasa_percentcut
    
    if CG_type=='CA':
        pred = pred
    if CG_type=='CACB':
        pred = np.sum(pred, axis=1)
        biodf_data = PandasPdb().read_pdb(pdb_filename)
        df_atom = biodf_data.df['ATOM']
        df_atom = df_atom[df_atom['chain_id'] == df_atom['chain_id'].unique()[0]]
        df_CA = df_atom[df_atom['atom_name'] == 'CA']
        resi_seq = df_CA['residue_name'].values
        for i in range(len(resi_seq)):
            if resi_seq[i]=='GLY':
                pred[i] = pred[i]/2
    if CG_type=='martini':
        biodf_data = PandasPdb().read_pdb(pdb_filename)
        df_atom = biodf_data.df['ATOM']
        df_atom = df_atom[df_atom['chain_id'] == df_atom['chain_id'].unique()[0]]
        df_atom['sasa'] = pred
        pred = np.array([np.sum(df_atom[df_atom['residue_number']==resi]['sasa']) for resi in df_atom['residue_number'].unique()])
    if CG_type=='P':
        pred = pred
    if CG_type=='3SPN':
        pred = np.sum(pred, axis=1)   
    return pred

def naccess(pdb_filename):
    '''
    use NACCESS to give a prediction
    '''
    delmkdir('./temp')
    naccess = NACCESS.run_naccess(biopy_Model.Model(0), pdb_filename, temp_path='./temp')
    resi_naccess_raw = naccess[0][4:-4]
    resi_naccess = np.array([float(raw[16:22]) for raw in resi_naccess_raw])
    shutil.rmtree('./temp')
    return resi_naccess

def output_sasa(pred, pdb_filename, output_filename):
    '''
    write prediction to a csv file
    '''
    df_atom = PandasPdb().read_pdb(pdb_filename).df['ATOM']
    df_atom = df_atom[df_atom['chain_id'] == df_atom['chain_id'].unique()[0]]
    resi_idx = df_atom['residue_number'].unique()
    resi_name = [df_atom[df_atom['residue_number'] == resi]['residue_name'].values[0] for resi in resi_idx]
    df_sasa=pd.DataFrame({'residue_number':resi_idx, 'residue_name':resi_name, 'SASA':pred})
    df_sasa.to_csv(output_filename+'.csv', index=False)
    
def main():
    '''
    get user input
    '''
    parser = optparse.OptionParser()
    parser.add_option('-f', '--file', dest='pdb_filename', default = './example_CA.pdb', help='The PDB file containing a CG protein or RNA structure as discussed in paper. Please use same format as shown in example input files (pdb format): example_CA.pdb, example_CACB.pdb ......')
    parser.add_option('-t', '--CG_type', dest='CG_type', default='CA', help='CG type of the PDB file, available for CA (Cα protein structure), CACB (Cα-Cβ protein structure), martini (Martini protein strcture), P (P-based RNA structure), 3SPN (3SPN RNA structure), AA (all-atom structure, calculate by NACCESS). For example: \"python deepCGSA.py -f example_CA.pdb -t CA -o output_filename\"')
    parser.add_option('-o', '--output', dest='output_filename', default='./output.csv', help='Residue-wise prediction will write to a csv file named as output_filename.csv')
    parser.add_option('-c', '--create_CG_file', dest='create_CG', default=False, help='We provided a convenient function to create CG file with appropriate format from AA file. -c option gave the CG type to convert. When using -c option, script will not calculate SASA but output a CG file. For example: \"python DeepCGSA.py -f example_AA.pdb -c CA\". Created file will be named as xxx_CGtype.pdb')
    options,args=parser.parse_args()
    check_input(options)
    
    pdb_filename = options.pdb_filename
    CG_type = options.CG_type
    output_filename = options.output_filename
    create_CG = options.create_CG
    
    if create_CG!=False:
        create_testCGfile(pdb_filename, create_CG)
    else:
        if CG_type in CG_types:
            model = DeepCGSA(CG_type)
            model.initial_model()
            bead_onehot, dismap = create_input_array(pdb_filename, CG_type)
            pred = pred_sasa(model, bead_onehot, dismap, pdb_filename, CG_type)
        else:
            pred = naccess(pdb_filename)

        output_sasa(pred, pdb_filename, output_filename)
    print('DONE!')

if __name__=='__main__':
    main()
