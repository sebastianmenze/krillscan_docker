

import shutil
from skimage.transform import  resize

from echolab2.instruments import EK80, EK60
import configparser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os

# from scipy.ndimage.filters import uniform_filter1d

from scipy.signal import convolve2d
# from skimage.transform import  resize

from echopy import transform as tf
from echopy import resample as rs
from echopy import mask_impulse as mIN
from echopy import mask_seabed as mSB
from echopy import get_background as gBN
from echopy import mask_range as mRG
from echopy import mask_shoals as mSH
from echopy import mask_signal2noise as mSN

from pyproj import Geod
geod = Geod(ellps="WGS84")
from pathlib import Path


# from matplotlib.colors import ListedColormap
import re
import traceback
# from pyproj import Proj, transform
import zipfile

import smtplib
import ssl
# import mimetypes
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.text  import MIMEText

from threading import Timer

        
#%%


class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

class krillscan():
    
    def start(self):
        print('start')
        self.callback_process_active=False
        self.callback_email_active=False
        # self.callback_plot_active=False
        self.df_files=pd.DataFrame([])
        self.echogram=pd.DataFrame([])    
        self.positions=pd.DataFrame([])    

        
        # # if os.path.isfile('settings.ini'):
        # config = configparser.ConfigParser()
        # config.read('settings.ini') 
        # config.items('GENERAL')
        # # else:
        # #     print('back to default settings')
        # #     self.settings_default()
           
        # self.folder_source=  str(config['GENERAL']['source_folder'])
        # self.folder_target=  str(config['GENERAL']['target_folder'])
        # w.pass_source_folder(folder_source)   
        # w_email.pass_source_folder(folder_source)   
        
        # fn=os.path.basename(os.path.normpath(self.folder_source))
        # workpath=os.path.join( str( pathlib.Path().resolve() ), '/target_folder')
        # workpath=self.folder_target
        # print(workpath)
        
        
        # if not os.path.isdir(workpath):
        #     os.mkdir(workpath)
        
        # w.pass_work_folder(workpath)   
        # w_email.pass_work_folder(workpath)   
        
        # if not os.path.isfile(os.path.join(workpath,'settings.ini')):
        #     shutil.copy2('settings.ini', workpath) 
            
        self.workpath = 'target_folder'
        print(os.getcwd())
        
        # os.chdir(self.workpath)
        

        self.timer_process = RepeatTimer(1, self.callback_process_raw)
        self.timer_process.start()
        self.timer_email = RepeatTimer(3, self.callback_email)
        self.timer_email.start()  
        
        # doc.add_periodic_callback( self.callback_process_raw,1000 ) 
        # doc.add_periodic_callback( self.callback_plot,3000 ) 
        # doc.add_periodic_callback( self.callback_email,5000 ) 

                
    def stop(self):
        self.timer_process.cancel()
        self.timer_email.cancel()  

    # def settings_default():          
    #     config = configparser.ConfigParser()
    #     config['GENERAL'] = {'source_folder' : '/source_folder', 
    #                          'target_folder' : '/target_folder', 
    #                          'transducer_frequency': 120000.0,
    #                          'vessel_name': 'MS example'}
    #     config['CALIBRATION'] = {'gain': 'None',
    #                          'sa_correction': 'None',
    #                          'beam_width_alongship':'None',
    #                          'beam_width_athwartship':'None',
    #                          'angle_offset_alongship':'None',
    #                          'angle_offset_athwartship':'None'}
    #     config['GRAPHICS'] = {'sv_min': -80,
    #                          'sv_max': -40,
    #                          'min_speed_in_knots': 5,
    #                          'nasc_map_max': 10000,
    #                          'nasc_graph_max': 50000,
    #                          'last_n_minutes_echogram': 30,
    #                          'last_n_days_map': 10,
    #                          }
                             
    #     config['EMAIL'] = {'email_from': "raw2nasc@gmail.com",
    #                          'email_to': "raw2nasc@gmail.com",
    #                          'pw': "myxdledwtfwuezis",
    #                          'files_per_email': 6*4,
    #                          'send_echograms': False,
    #                          'echogram_resolution_in_seconds': 60}
       
    #     with open('settings.ini', 'w') as configfile:
    #       config.write(configfile)      

    def read_raw(self,rawfile):       
        df_sv=pd.DataFrame( [] )
        positions=pd.DataFrame( []  )
        
        # breakpoint()
        
        # print('Echsounder data are: ')
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')   
   
        try:     
            raw_obj = EK80.EK80()
            raw_obj.read_raw(rawfile)
            print(raw_obj)
        except Exception as e:            
            print(e)       
            try:     
                raw_obj = EK60.EK60()
                raw_obj.read_raw(rawfile)
                print(raw_obj)
            except Exception as e:
                print(e)       
                
                                           
        
        raw_freq= list(raw_obj.frequency_map.keys())
        
        # self.ekdata=dict()
        
        # for f in raw_freq:
        f=float(self.config['GENERAL']['transducer_frequency'])
        print(raw_freq)
     
        raw_data = raw_obj.raw_data[raw_obj.frequency_map[f][0]][0]  

        if np.shape(raw_data)[0]>1:                     
            cal_obj = raw_data.get_calibration()
            
            try: 
               cal_obj.gain=float(self.config['CALIBRATION']['gain']       )
            except:
                pass
            try: 
               cal_obj.sa_correction=float(self.config['CALIBRATION']['sa_correction']       )
            except:
                pass
            try: 
               cal_obj.beam_width_alongship=float(self.config['CALIBRATION']['beam_width_alongship']       )
            except:
                pass
            try: 
               cal_obj.beam_width_athwartship=float(self.config['CALIBRATION']['beam_width_athwartship']       )
            except:
                pass
            try: 
               cal_obj.angle_offset_alongship=float(self.config['CALIBRATION']['angle_offset_alongship']       )
            except:
                pass
            try: 
               cal_obj.angle_offset_athwartship=float(self.config['CALIBRATION']['angle_offset_athwartship']       )
            except:
                pass
                
            
            sv_obj = raw_data.get_sv(calibration = cal_obj)    
              
            positions =pd.DataFrame(  raw_obj.nmea_data.interpolate(sv_obj, 'GGA')[1] )
           
            svr = np.transpose( 10*np.log10( sv_obj.data ) )
            
            # print(svr)
    
           
            # r=np.arange( sv_obj.range.min() , sv_obj.range.max() , 0.5 )
            r=np.arange( 0 , sv_obj.range.max() , 0.5 )
    
            t=sv_obj.ping_time
    
            sv=  resize(svr,[ len(r) , len(t) ] )
    
           # print(sv.shape)
           
            # estimate and correct background noise       
            p         = np.arange(len(t))                
            s         = np.arange(len(r))          
            bn, m120bn_ = gBN.derobertis(sv, s, p, 5, 20, r, np.mean(cal_obj.absorption_coefficient) ) # whats correct absoprtion?
            b=pd.DataFrame(bn)
            bn=  b.interpolate(axis=1).interpolate(axis=0).values                        
            sv_clean     = tf.log(tf.lin(sv) - tf.lin(bn))
    
         # -------------------------------------------------------------------------
         # mask low signal-to-noise 
            msn             = mSN.derobertis(sv_clean, bn, thr=12)
            sv_clean[msn] = np.nan
    
        # get mask for seabed
            mb = mSB.ariza(sv, r, r0=20, r1=1000, roff=0,
                              thr=-38, ec=1, ek=(3,3), dc=10, dk=(5,15))
            sv_clean[mb]=-999
                                 
            df_sv=pd.DataFrame( np.transpose(sv_clean) )
            df_sv.index=t
            df_sv.columns=r
            
            # print(df_sv)
            # print(positions)
               
            return df_sv, positions

    def callback_process_raw(self):
              
      config = configparser.ConfigParser()
      config.read('settings.ini')            
      self.folder_source=  str(config['GENERAL']['source_folder'])  
        
      if (self.callback_process_active==False) :
          
        self.callback_process_active==True
        # self.workpath=  os.path.join(self.folder_source,'krill_data')     
        # os.chdir(self.workpath)
    
        new_df_files = pd.DataFrame([])           
        new_df_files['path'] = glob.glob( os.path.join( self.folder_source,'*.raw') )  
        print('found '+str(len(new_df_files)) + ' raw files')
    
        dates=[]
        for fname in new_df_files['path']:
            
            datetimestring=re.search('D\d\d\d\d\d\d\d\d-T\d\d\d\d\d\d',fname).group()
            dates.append( pd.to_datetime( datetimestring,format='D%Y%m%d-T%H%M%S' ) )
        new_df_files['date'] = dates
    
    
        new_df_files['to_do']=True 
        
        
        self.df_files=pd.concat([self.df_files,new_df_files])
        self.df_files.drop_duplicates(inplace=True)
        
        self.df_files =  self.df_files.sort_values('date')
        self.df_files=self.df_files.reset_index(drop=True)
        
     
        
        # look for already processed data
        self.df_files['to_do']=True    
        
        if os.path.isfile(self.workpath+'/list_of_rawfiles.csv'):
            df_files_done =  pd.read_csv(self.workpath+'/list_of_rawfiles.csv',index_col=0)
            df_files_done=df_files_done.loc[ df_files_done['to_do']==False,: ]
        
            names = self.df_files['path'].apply(lambda x: Path(x).stem)       
            names_done = df_files_done['path'].apply(lambda x: Path(x).stem)       
            
        # print(names)
        # print(nasc_done)
            ix_done= names.isin( names_done  )  
    
        # print(ix_done)
            self.df_files.loc[ix_done,'to_do'] = False        
        self.n_todo=np.sum(self.df_files['to_do'])
        print('To do: ' + str(self.n_todo))
        
        # echogram=pd.DataFrame([])    
        # positions=pd.DataFrame([])    
        
        unit_length_min=pd.to_timedelta(10,'min')
        
        ix_todo = np.where( self.df_files['to_do']==True )[0]
        if self.n_todo>0:
                index = ix_todo[0]
                row = self.df_files.iloc[ index ,:]

        # for index, row in self.df_files.iterrows():
        #     if self.toggle_proc.active & (row['to_do']==True):
                rawfile=row['path']
                print('working on '+rawfile)
                try:
                    
                    # breakpoint()
                    
                    echogram_file, positions_file = self.read_raw(rawfile)
                    
                    
                    self.echogram = pd.concat([ self.echogram,echogram_file ])
                    self.positions = pd.concat([ self.positions,positions_file ])
                    t=self.echogram.index
                    
                    # print(echogram)
                    
                    # print( [ t.max() , t.min() ])
                    
                    while (t.max() - t.min()) > unit_length_min:
                        
                        # print(  (t.min() + unit_length_min) > t)
                        ix_end = np.where( (t.min() + unit_length_min) > t )[0][-1]
                        ix_start=t.argmin()
                        # print([ix_start,ix_end])
                        
                        # accumulate 10 min snippet  
                        new_echogram = self.echogram.iloc[ix_start:ix_end,:]
                        new_positions = self.positions.iloc[ix_start:ix_end,:]
                        self.echogram = self.echogram.iloc[ix_end::,:]
                        self.positions = self.positions.iloc[ix_end::,:]
                        t=self.echogram.index

                        # try:
                        df_nasc_file, df_sv_swarm = self.detect_krill_swarms(new_echogram,new_positions)   
                        name = t.min().strftime('D%Y%m%d-T%H%M%S' )         
                        
                        df_sv_swarm[ new_echogram==-999 ] =-999
                        
                                                
                        df_nasc_file.to_hdf(self.workpath+'/'+ name + '_nasctable.h5', key='df', mode='w'  )
                        
                        dffloat=df_nasc_file.copy()
                        formats = {'lat': "{:.6f}", 'lon': "{:.6f}", 'distance_m': "{:.4f}",'bottomdepth_m': "{:.1f}",'nasc': "{:.2f}"}
                        for col, f in formats.items():
                            dffloat[col] = dffloat[col].map(lambda x: f.format(x))                           
                        # dffloat.to_csv( name + '_nasctable.gzip',compression='gzip' )
                        dffloat.to_csv(self.workpath+'/'+ name + '_nasctable.csv')
                        
                        df_sv_swarm.astype('float16').to_hdf(self.workpath+'/'+ name + '_sv_swarm.h5', key='df', mode='w'  )
                        # self.df_files.loc[i,'to_do'] = False
                        # except Exception as e:
                        #   print(e)                      
                    self.df_files.loc[index,'to_do']=False            
                    self.df_files.drop_duplicates(inplace=True)
                    self.df_files=self.df_files.reset_index(drop=True)
                    self.df_files.to_csv(self.workpath+'/list_of_rawfiles.csv')
                   
                except Exception as e:
                    print(e)               
                    print(traceback.format_exc())
                    # breakpoint()
        self.callback_process_active==False
                
    def detect_krill_swarms(self,sv,positions):
         # sv= self.echodata[rawfile][ 120000.0] 
         # sv= self.ekdata[ 120000.0]          
         # breakpoint()
              
         t120 =sv.index
         r120 =sv.columns.values

         Sv120=  np.transpose( sv.values )
         # get swarms mask
         k = np.ones((3, 3))/3**2
         Sv120cvv = tf.log(convolve2d(tf.lin( Sv120 ), k,'same',boundary='symm'))   
 
         p120           = np.arange(np.shape(Sv120cvv)[1]+1 )                 
         s120           = np.arange(np.shape(Sv120cvv)[0]+1 )           
         m120sh, m120sh_ = mSH.echoview(Sv120cvv, s120, p120, thr=-70,
                                    mincan=(3,10), maxlink=(3,15), minsho=(3,15))

        # -------------------------------------------------------------------------
        # get Sv with only swarms
         Sv120sw =  Sv120.copy()
         Sv120sw[~m120sh] = np.nan
  
         ixdepthvalid= (r120>=20) & (r120<=500)
         Sv120sw[~ixdepthvalid,:]=np.nan
  
         
         cell_thickness=np.abs(np.mean(np.diff( r120) ))               
         nasc_swarm=4*np.pi*1852**2 * np.nansum( np.power(10, Sv120sw /10)*cell_thickness ,axis=0)   
         
         # nasc_swarm[nasc_swarm>20000]=np.nan
                         
          
         df_sv_swarm=pd.DataFrame( np.transpose(Sv120sw) )
         df_sv_swarm.index=t120
         df_sv_swarm.columns=r120
          # print('df_sv')
         
         df_nasc_file=pd.DataFrame([])
         # df_nasc_file['time']=positions['ping_time']
         df_nasc_file['lat']=positions['latitude']
         df_nasc_file['lon']=positions['longitude']
         df_nasc_file['distance_m']=np.append(np.array([0]),geod.line_lengths(lons=positions['longitude'],lats=positions['latitude']) )
         
         bottomdepth=[]
         for index_1, row_1 in sv.iterrows():
             if np.sum(row_1==-999)>0:
                 bottomdepth.append( np.min(r120[row_1==-999]) )
             else:
                 bottomdepth.append( r120.max() )
            
         df_nasc_file['bottomdepth_m']=bottomdepth
            
           
         df_nasc_file['nasc']=nasc_swarm
         df_nasc_file.index=positions['ping_time']
         
         # df_nasc_file=df_nasc_file.resample('5s').mean()
         print('Krill detection complete: '+str(np.sum(nasc_swarm)) ) 
        
         return df_nasc_file, df_sv_swarm
        
         
    # def start():
    #     print('start')

    def callback_email(self):
      if  (self.callback_email_active==False) :      
        self.callback_email_active==True
        print('checking wether to send email')
        self.config = configparser.ConfigParser()
        self.config.read('settings.ini')   
                
        emailfrom = self.config['EMAIL']['email_from']
        emailto = self.config['EMAIL']['email_to']
        # fileToSend = r"D20220212-T180420_nasctable.h5"
        # username = "raw2nasc"
        # password = "raw2nasckrill"
        password =self.config['EMAIL']['pw']
        
        # breakpoint()
        
        # self.workpath=  os.path.join(self.folder_source,'krill_data')
        
        # os.chdir(self.workpath)
        # self.df_files=pd.read_csv(self.workpath+'/list_of_rawfiles.csv')
       
        nasc_done =  pd.DataFrame( glob.glob( self.workpath+'/*_nasctable.h5' ) )
        if len(nasc_done)>0:               
            if os.path.isfile(self.workpath+'/list_of_sent_files.csv'):
                df_files_sent =  pd.read_csv(self.workpath+'/list_of_sent_files.csv',index_col=0)
                ix_done= nasc_done.iloc[:,0].isin( df_files_sent.iloc[:,0]  )  
                nasc_done=nasc_done[~ix_done]
            
            else:    
                df_files_sent=pd.DataFrame([])
            
            nascfile_times=[]
            for fname in nasc_done.iloc[:,0]:         
                datetimestring=re.search('D\d\d\d\d\d\d\d\d-T\d\d\d\d\d\d',fname).group()
                nascfile_times.append( pd.to_datetime( datetimestring,format='D%Y%m%d-T%H%M%S' ) )
            
            # nascfile_times=pd.to_datetime( nasc_done.iloc[:,0] ,format='D%Y%m%d-T%H%M%S_nasctable.h5' )
            nasc_done=nasc_done.iloc[np.argsort(nascfile_times),0].values
                 
            n_files=int(self.config['EMAIL']['files_per_email'])
            send_echograms=bool(self.config['EMAIL']['send_echograms'])
            echogram_resolution_in_seconds=str(self.config['EMAIL']['echogram_resolution_in_seconds'])
            print( str(len(nasc_done)) +' files that can be sent')

            while (len(nasc_done)>n_files) :
                
                
                files_to_send=nasc_done[0:n_files]
                # print(nasc_done)
                
                msg = MIMEMultipart()
                msg["From"] = emailfrom
                msg["To"] = emailto
                msg["Subject"] = "Krillscan data from "+ self.config['GENERAL']['vessel_name']+' ' +files_to_send[0][-30:-13]+'_to_'+files_to_send[-1][-30:-13]
              
                msgtext = str(dict(self.config['GENERAL']))
                msg.attach(MIMEText( msgtext   ,'plain'))

                loczip = msg["Subject"]+'.zip'
                zip = zipfile.ZipFile(loczip, "w", zipfile.ZIP_DEFLATED)
                zip.write('settings.ini')

                for fi in files_to_send:   
                    zip.write(fi,arcname=fi[-30:]  )                                  

                if send_echograms:                       
                    for fi in files_to_send:      

                        # fi=        files_to_send.iloc[0,0]
                        df = pd.read_hdf(self.workpath+'/'+fi[-30:-13] + '_sv_swarm.h5' ,key='df') 
                        df=df.resample(echogram_resolution_in_seconds+'s').mean()
                        targetname=fi[-30:-13] + '_sv_swarm_mail.h5' 
                        df.astype('float16').to_hdf(targetname,key='df',mode='w')
                        # df.astype('float16').to_csv(targetname,compression='gzip')
                        zip.write(targetname)                                                      
                        os.remove(targetname)
                
                zip.close()
                fp = open(loczip, "rb")
                attachment = MIMEBase('application', 'x-zip')
                attachment.set_payload(fp.read())
                fp.close()
                encoders.encode_base64(attachment)
                attachment.add_header("Content-Disposition", "attachment", filename=loczip)
                msg.attach(attachment)    
                
                os.remove(loczip)

                try:        
                    ctx = ssl.create_default_context()
                    server = smtplib.SMTP_SSL("smtp.gmail.com", port=465, context=ctx)
                    
                    server.login(emailfrom, password)
                    
                    # print(df_files_sent)
                
                    server.sendmail(emailfrom, emailto.split(','), msg.as_string())
                    if len(df_files_sent)>0:
                        df_files_sent= pd.concat([pd.Series(df_files_sent.iloc[:,0].values),pd.DataFrame(files_to_send)],ignore_index=True)
                    else:
                        df_files_sent=pd.DataFrame(files_to_send)
                        
                    # df_files_sent=df_files_sent.reset_index(drop=True)
                    df_files_sent=df_files_sent.drop_duplicates()
                    df_files_sent.to_csv(self.workpath+'/list_of_sent_files.csv')
                    
                    
                    print('email sent: ' +   msg["Subject"] )
                    nasc_done=nasc_done[n_files::]
                    server.quit()

                except Exception as e:
                    print(e)
                                        
        
        self.callback_email_active==False
        
#%%

ks=krillscan()
ks.start()


# ks.stop()