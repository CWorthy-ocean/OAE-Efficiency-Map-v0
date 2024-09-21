#!/bin/bash

JobIds=$(squeue -u ${USER} | grep "${USER} PD" | awk '{print $1}')


for JobId in ${JobIds}; do
   echo ${JobId}
   scontrol update JobId=${JobId} ReservationName=cdr_atlas2
done