#! /usr/bin/python

######################################
#                                    #
# Program for collecting information #
# from a set of SLHA files           #
#                                    #
######################################
#
# Usage:
# ------
#     python harvest_slha_ew.py <output file> <root search dir> <file tag>
#
#
# Details:
# --------
# The program will only include files that have <file tag> as part of the filename.
# If <file tag> is set to '', all files in the search directory are included.
#
#

import os
import sys
#from modules import pyslha
import pyslha
from collections import OrderedDict


###########################
#  What data to collect?  #
###########################

# Should this really be ordered? Try to change
datadict = OrderedDict ([])

# The element stored in the dictionary is defined by a pair (key,index).
# The key can itself be a tuple (n1,n2,...) if the SLHA block has multiple
# indices. The index is the column number of the sought entry on the line of the
# SLHA file counting from 0 and *including* the key entries.

# MSSM parameters
datadict['M1'] = {'block': 'EXTPAR',     'element': (1,1),   'abs' : False}
datadict['M2'] = {'block': 'EXTPAR',     'element': (2,1),   'abs' : False}
datadict['mu'] = {'block': 'EXTPAR',     'element': (23,1),   'abs' : False}
datadict['tanb'] = {'block': 'MINPAR',     'element': (3,1),   'abs' : True}


# Chargino and neutralino masses
# TODO: check if we really want abs
datadict['m1000022'] = {'block': 'MASS',     'element': (1000022,1),   'abs' : True}
datadict['m1000023'] = {'block': 'MASS',     'element': (1000023,1),   'abs' : True}
datadict['m1000024'] = {'block': 'MASS',     'element': (1000024,1),   'abs' : True}
datadict['m1000025'] = {'block': 'MASS',     'element': (1000025,1),   'abs' : True}
datadict['m1000035'] = {'block': 'MASS',     'element': (1000035,1),   'abs' : True}
datadict['m1000037'] = {'block': 'MASS',     'element': (1000037,1),   'abs' : True}

# Other relevant masses
#datadict['m1000021'] = {'block': 'MASS',     'element': (1000021,1),   'abs' : True}
#datadict['m1000004'] = {'block': 'MASS',     'element': (1000004,1),   'abs' : True}
#datadict['m1000003'] = {'block': 'MASS',     'element': (1000003,1),   'abs' : True}
#datadict['m1000001'] = {'block': 'MASS',     'element': (1000001,1),   'abs' : True}
#datadict['m1000002'] = {'block': 'MASS',     'element': (1000002,1),   'abs' : True}
#datadict['m2000002'] = {'block': 'MASS',     'element': (2000002,1),   'abs' : True}
#datadict['m2000001'] = {'block': 'MASS',     'element': (2000001,1),   'abs' : True}
#datadict['m2000003'] = {'block': 'MASS',     'element': (2000003,1),   'abs' : True}
#datadict['m2000004'] = {'block': 'MASS',     'element': (2000004,1),   'abs' : True}

# Neutralino mixing matrix
datadict['nmix11'] = {'block': 'NMIX', 'element': ((1,1),2)}
datadict['nmix12'] = {'block': 'NMIX', 'element': ((1,2),2)}
datadict['nmix13'] = {'block': 'NMIX', 'element': ((1,3),2)}
datadict['nmix14'] = {'block': 'NMIX', 'element': ((1,4),2)}
datadict['nmix21'] = {'block': 'NMIX', 'element': ((2,1),2)}
datadict['nmix22'] = {'block': 'NMIX', 'element': ((2,2),2)}
datadict['nmix23'] = {'block': 'NMIX', 'element': ((2,3),2)}
datadict['nmix24'] = {'block': 'NMIX', 'element': ((2,4),2)}


# Do decays
decays = True

if decays:
    try:
        datadict['br1000023_12'] = {'decay': 1000023, 'ids': [1000022, 12, -12] }
        datadict['br1000023_23'] = {'decay': 1000023, 'ids': [1000022, 23] }
        datadict['br1000023_25'] = {'decay': 1000023, 'ids': [1000022, 25] }
    except:
        pass


# Chargino mixing matrix
code=''
#code = 'resummino'
#code = 'prospino'
#com_energies = [7000, 8000, 13000, 14000]
com_energies = [13000]

# Loop over energies
for en in com_energies:

    # Reading resummino output
    if code == 'resummino':
        try:
            # chi01 chi01
#            datadict['1000022_1000022_' + str(en) + '_LO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000022, 1000022, en), 3)}
#            datadict['1000022_1000022_' + str(en) + '_LO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000022, 1000022, en), 4)}
#            datadict['1000022_1000022_' + str(en) + '_NLO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000022, 1000022, en), 5)}
#            datadict['1000022_1000022_' + str(en) + '_NLO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000022, 1000022, en), 6)}
#            datadict['1000022_1000022_' + str(en) + '_NLL'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000022, 1000022, en), 7)}
#            datadict['1000022_1000022_' + str(en) + '_NLL_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000022, 1000022, en), 8)}
            # chi02 chi02
            datadict['1000023_1000023_' + str(en) + '_LO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000023, en), 3)}
            datadict['1000023_1000023_' + str(en) + '_LO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000023, en), 4)}
            datadict['1000023_1000023_' + str(en) + '_NLO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000023, en), 5)}
            datadict['1000023_1000023_' + str(en) + '_NLO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000023, en), 6)}
            datadict['1000023_1000023_' + str(en) + '_NLL'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000023, en), 7)}
            datadict['1000023_1000023_' + str(en) + '_NLL_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000023, en), 8)}
            # chi02 chi1+
            datadict['1000023_1000024_' + str(en) + '_LO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000024, en), 3)}
            datadict['1000023_1000024_' + str(en) + '_LO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000024, en), 4)}
            datadict['1000023_1000024_' + str(en) + '_NLO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000024, en), 5)}
            datadict['1000023_1000024_' + str(en) + '_NLO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000024, en), 6)}
            datadict['1000023_1000024_' + str(en) + '_NLL'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000024, en), 7)}
            datadict['1000023_1000024_' + str(en) + '_NLL_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, 1000024, en), 8)}
            # chi02 chi1-
            datadict['1000023_-1000024_' + str(en) + '_LO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, -1000024, en), 3)}
            datadict['1000023_-1000024_' + str(en) + '_LO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, -1000024, en), 4)}
            datadict['1000023_-1000024_' + str(en) + '_NLO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, -1000024, en), 5)}
            datadict['1000023_-1000024_' + str(en) + '_NLO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, -1000024, en), 6)}
            datadict['1000023_-1000024_' + str(en) + '_NLL'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, -1000024, en), 7)}
            datadict['1000023_-1000024_' + str(en) + '_NLL_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000023, -1000024, en), 8)}
            # chi1+ chi1-
            datadict['1000024_-1000024_' + str(en) + '_LO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000024, -1000024, en), 3)}
            datadict['1000024_-1000024_' + str(en) + '_LO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000024, -1000024, en), 4)}
            datadict['1000024_-1000024_' + str(en) + '_NLO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000024, -1000024, en), 5)}
            datadict['1000024_-1000024_' + str(en) + '_NLO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000024, -1000024, en), 6)}
            datadict['1000024_-1000024_' + str(en) + '_NLL'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000024, -1000024, en), 7)}
            datadict['1000024_-1000024_' + str(en) + '_NLL_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000024, -1000024, en), 8)}
            # chi03 chi1+
            datadict['1000025_1000024_' + str(en) + '_LO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, 1000024, en), 3)}
            datadict['1000025_1000024_' + str(en) + '_LO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, 1000024, en), 4)}
            datadict['1000025_1000024_' + str(en) + '_NLO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, 1000024, en), 5)}
            datadict['1000025_1000024_' + str(en) + '_NLO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, 1000024, en), 6)}
            datadict['1000025_1000024_' + str(en) + '_NLL'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, 1000024, en), 7)}
            datadict['1000025_1000024_' + str(en) + '_NLL_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, 1000024, en), 8)}
            # chi03 chi1-
            datadict['1000025_-1000024_' + str(en) + '_LO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, -1000024, en), 3)}
            datadict['1000025_-1000024_' + str(en) + '_LO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, -1000024, en), 4)}
            datadict['1000025_-1000024_' + str(en) + '_NLO'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, -1000024, en), 5)}
            datadict['1000025_-1000024_' + str(en) + '_NLO_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, -1000024, en), 6)}
            datadict['1000025_-1000024_' + str(en) + '_NLL'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, -1000024, en), 7)}
            datadict['1000025_-1000024_' + str(en) + '_NLL_err'] = {'block': 'RESUMMINO_OUTPUT', 'element': ((1000025, -1000024, en), 8)}

        except:
            pass

    # Reading Prospino output
    if code == 'prospino':
        try:
            # chi01-chi01 cross section and prospino numerical error
            datadict['1000022_1000022_' + str(en) + '_NLO_1_relerr'] = {'block': 'PROSPINO_OUTPUT', 'element': ((1000022, 1000022, en),6)}
            datadict['1000022_1000022_' + str(en) + '_NLO_1']        = {'block': 'PROSPINO_OUTPUT', 'element': ((1000022, 1000022, en),7)}
            # Scale variations, 0.5 and 2 times the central scale
            datadict['1000022_1000022_' + str(en) + '_NLO_05']     = {'block': 'PROSPINO_OUTPUT', 'element': ((1000022, 1000022, en),8)}
            datadict['1000022_1000022_' + str(en) + '_NLO_2']      = {'block': 'PROSPINO_OUTPUT', 'element': ((1000022, 1000022, en),9)}
            # PDF variation
            datadict['1000022_1000022_' + str(en) + '_NLO_pdf']    = {'block': 'PROSPINO_OUTPUT', 'element': ((1000022, 1000022, en),10)}
            # Alpha_s variations, up and down one sigma (PDG)
            datadict['1000022_1000022_' + str(en) + '_NLO_aup']    = {'block': 'PROSPINO_OUTPUT', 'element': ((1000022, 1000022, en),11)}
            datadict['1000022_1000022_' + str(en) + '_NLO_adn']    = {'block': 'PROSPINO_OUTPUT', 'element': ((1000022, 1000022, en),12)}

            # chi02-chi02 cross section and prospino numerical error
            datadict['1000023_1000023_' + str(en) + '_NLO_1_relerr'] = {'block': 'PROSPINO_OUTPUT', 'element': ((1000023, 1000023, en),6)}
            datadict['1000023_1000023_' + str(en) + '_NLO_1']        = {'block': 'PROSPINO_OUTPUT', 'element': ((1000023, 1000023, en),7)}
            # Scale variations, 0.5 and 2 times the central scale
            datadict['1000023_1000023_' + str(en) + '_NLO_05']     = {'block': 'PROSPINO_OUTPUT', 'element': ((1000023, 1000023, en),8)}
            datadict['1000023_1000023_' + str(en) + '_NLO_2']      = {'block': 'PROSPINO_OUTPUT', 'element': ((1000023, 1000023, en),9)}
            # PDF variation
            datadict['1000023_1000023_' + str(en) + '_NLO_pdf']    = {'block': 'PROSPINO_OUTPUT', 'element': ((1000023, 1000023, en),10)}
            # Alpha_s variations, up and down one sigma (PDG)
            datadict['1000023_1000023_' + str(en) + '_NLO_aup']    = {'block': 'PROSPINO_OUTPUT', 'element': ((1000023, 1000023, en),11)}
            datadict['1000023_1000023_' + str(en) + '_NLO_adn']    = {'block': 'PROSPINO_OUTPUT', 'element': ((1000023, 1000023, en),12)}
        except:
            pass


##############################
#  Initial setup and checks  #
##############################

# set output prefix
outpref = sys.argv[0] + ' : '

# check input arguments:
if len(sys.argv) != 4:
    sys.stdout.write("%s Wrong number of input arguments.\n" % (outpref))
    sys.stdout.write("%s Usage:\n" % (outpref))
    sys.stdout.write("%s   python harvest_slha_nimbus.py <output file> <root search dir> <file tag>\n" % (outpref))
    sys.exit()

# assign input arguments to variables
outfile = sys.argv[1]
searchdir = sys.argv[2]
filetag = sys.argv[3]


#####################################
#  File search and data collection  #
#####################################

inputfiles = []
for root, dirnames, filenames in os.walk(searchdir):
    # for filename in fnmatch.filter(filenames, filetag):
    for filename in filenames:

        if filetag not in filename:
            continue

        inputfiles.append(os.path.join(root, filename))


# print file count
n_files_total = len(inputfiles)
sys.stdout.write("%s Found %d input files.\n" % (outpref, n_files_total))

# sort the file list
inputfiles.sort()

# print info
sys.stdout.write("%s Collecting data...\n" % (outpref))

# open outfile for output
f = open(outfile, "w")

# add tag for filename columns
max_path_lenght = len(max(filenames))
tagline = ''
tag = 'file'
tagline += ( tag+' '*(max_path_lenght-4) )

for i, tag in enumerate(datadict.keys()): # IH: Remove the tag for masses and xsections
    n = i+2
    complete_tag = tag # IH
    #complete_tag = '%i.%s' % (n,tag)
    tagline += ( complete_tag+' '*(max(1,14-len(complete_tag))) )


tagline += '\n'

f.write(tagline)

# collect data from each file and write to outfile
lines = ''
for count, filepath in enumerate(inputfiles, 1):

    slha_dict = pyslha.readSLHAFile(filepath)

    datalist = []

    accepted_file = True
    number_of_unaccepted_sxections = 0 #IH

    for key in datadict.keys():
        if 'block' in datadict[key].keys():
            if not datadict[key]['block'] in slha_dict.keys():
                accepted_file = False
                sys.stdout.write("%s Problem encountered when looking for block %s in file %s. File ignored.\n" % (outpref, datadict[key]['block'], filepath))
                break
            if ('abs' in datadict[key].keys()) and (datadict[key]['abs'] == True):
                datalist.append( abs( slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] ) ) )
            else:
                try:
                    slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] ) #IH
                    #print slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] )
                    datalist.append( slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] ) )
                except:
                    number_of_unaccepted_sxections += 1
                    datalist.append(-1)

        elif 'decay' in datadict[key].keys():
            #print datadict[key]['ids']
            if not datadict[key]['decay'] in slha_dict.keys():
                accepted_file = False
                sys.stdout.write("%s Problem encountered when looking for decay %s in file %s. File ignored.\n" % (outpref, datadict[key]['decay'], filepath))
                break
            print(slha_dict[ datadict[key]['decay'] ])
            #print slha_dict[ datadict[key]['decay'] ].totalwidth
            #datalist.append( slha_dict[ datadict[key]['decay'] ].getBR( list( datadict[key]['element'] ) ) )
            datalist.append( slha_dict[ datadict[key]['decay'] ].getBR( datadict[key]['ids'] ) )


        # got_it = True
        # try:
        #     datalist.append( slha_dict[ datadict[key]['block'] ].getElement( *datadict[key]['element'] ) )

        # except Exception as err:
        #     sys.stdout.write("%s %s \n" % (outpref, err.message))
        #     sys.stdout.write("%s Problem encountered when harvesting data from file %s. File ignored.\n" % (outpref, filepath))
        #     continue

    if not accepted_file:
        continue


    datatuple = tuple(datalist)

    # Chop filepath for printing
    file = filepath.split('/')[-1]

    # Make lines for printing
    lines += ('%s' + ' '*(max_path_lenght-len(filepath)+2) ) % file
    lines += ( ('% .5e' + ' '*2)*len(datatuple) + '\n') % datatuple

    # write to file once per 1000 files read
    if count%1000 == 0:
        sys.stdout.write("%s %d of %d files read\n" % (outpref, count, n_files_total))
        f.write(lines)
        lines = ''


# Remove final endline and write remaining lines
lines = lines.rstrip('\n')
f.write(lines)

##############
#  Finalise  #
##############

# output info
sys.stdout.write("%s ...done!\n" % (outpref))

# close the outfile
f.close()

# print some output
sys.stdout.write("%s Summary written to the file %s \n" % (outpref, outfile))
