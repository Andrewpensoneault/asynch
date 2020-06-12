#ifndef FORCINGS_H
#define FORCINGS_H

#include <stdio.h>
#include <stdlib.h>
#include "structs.h"
#include "rainfall.h"
#include "io.h"

extern int np;
extern int my_rank;

Forcing* InitializeForcings();
void FreeForcing(Forcing** forcings);
void RefreshForcing(Forcing** forcings);

unsigned int PassesOther(Forcing* forcing,double maxtime,ConnData* conninfo);
unsigned int PassesBinaryFiles(Forcing* forcing,double maxtime,ConnData* conninfo);
unsigned int PassesDatabase(Forcing* forcing,double maxtime,ConnData* conninfo);
unsigned int PassesRecurring(Forcing* forcing,double maxtime,ConnData* conninfo);
unsigned int PassesDatabase_Irregular(Forcing* forcing,double maxtime,ConnData* conninfo, MPI_Comm comm);

double NextForcingOther(Link** sys,unsigned int N,unsigned int* my_sys,unsigned int my_N,int* assignments,UnivVars* GlobalVars,Forcing* forcing,ConnData** db_connections,unsigned int** id_to_loc,unsigned int forcing_idx);
double NextForcingBinaryFiles(Link** sys,unsigned int N,unsigned int* my_sys,unsigned int my_N,int* assignments,UnivVars* GlobalVars,Forcing* forcing,ConnData** db_connections,unsigned int** id_to_loc,unsigned int forcing_idx,MPI_Comm comm);
double NextForcingGZBinaryFiles(Link** sys,unsigned int N,unsigned int* my_sys,unsigned int my_N,int* assignments,UnivVars* GlobalVars,Forcing* forcing,ConnData** db_connections,unsigned int** id_to_loc,unsigned int forcing_idx, MPI_Comm comm);
double NextForcingGridCell(Link** sys,unsigned int N,unsigned int* my_sys,unsigned int my_N,int* assignments,UnivVars* GlobalVars,Forcing* forcing,ConnData** db_connections,unsigned int** id_to_loc,unsigned int forcing_idx,MPI_Comm comm);
double NextForcingDatabase(Link** sys,unsigned int N,unsigned int* my_sys,unsigned int my_N,int* assignments,UnivVars* GlobalVars,Forcing* forcing,ConnData** db_connections,unsigned int** id_to_loc,unsigned int forcing_idx,MPI_Comm comm);
double NextForcingRecurring(Link** sys,unsigned int N,unsigned int* my_sys,unsigned int my_N,int* assignments,UnivVars* GlobalVars,Forcing* forcing,ConnData** db_connections,unsigned int** id_to_loc,unsigned int forcing_idx,MPI_Comm comm);
double NextForcingDatabase_Irregular(Link** sys,unsigned int N,unsigned int* my_sys,unsigned int my_N,int* assignments,UnivVars* GlobalVars,Forcing* forcing,ConnData** db_connections,unsigned int** id_to_loc,unsigned int forcing_idx,MPI_Comm comm);

#endif

