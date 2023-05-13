#!/bin/csh -f

#------------------------------------------------------------------------------------
# For now, set streams manually. You must only set as many streams as are declared
#  in the tavg_nml section. For example, if there are three streams:
#  @ s1 = $my_stream
#  @ s2 = $s1 + 1
#  @ s3 = $s2 + 1
#------------------------------------------------------------------------------------
@ my_stream = $1
if ($my_stream < 1) then
   echo invalid my_stream number  ($my_stream)
   exit 5
endif

@ s1 = 1   # use base-model stream 1

set lmcog_debug = $2
set mcog_ncols  = $3
set mcog_nbins  = $4

if ($lmcog_debug == ".true.") then
   set stream_normal_var = "2"
   set stream_debug_var  = "2"
else
   set stream_normal_var = "1"
   set stream_debug_var  = "!"
endif

rm -f $CASEROOT/Buildconf/popconf/mcog_tavg_contents
touch $CASEROOT/Buildconf/popconf/mcog_tavg_contents
