#ifndef COMM_H
#define COMM_H

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "structs.h"
#include "system.h"
#include "sort.h"

extern int my_rank;
extern int np;

//MPI Related Methods
void Transfer_Data(TransData* my_data,Link** sys,int* assignments,UnivVars* GlobalVars,MPI_Comm comm);
void Transfer_Data_Finish(TransData* my_data,Link** sys,int* assignments,UnivVars* GlobalVars,MPI_Comm comm);
void Exchange_InitState_At_Forced(Link** system,unsigned int N,unsigned int* assignments,short int* getting,unsigned int* res_list,unsigned int res_size,unsigned int** id_to_loc,UnivVars* GlobalVars,MPI_Comm comm);
TransData* Initialize_TransData();
void Flush_TransData(TransData* data,MPI_Comm comm);
void TransData_Free(TransData* data,MPI_Comm comm);

//PostgreSQL Database Methods
ConnData* CreateConnData(char* connectstring);
void ConnData_Free(ConnData* conninfo);
void SwitchDB(ConnData* conninfo,char connectinfo[]);
int ConnectPGDB(ConnData* conninfo);
void DisconnectPGDB(ConnData* conninfo);
int CheckResError(PGresult* res,char* event);
int CheckResState(PGresult* res,short int error_code);
void CheckConnConnection(ConnData* conninfo);
void ShutUp(void *arg, const char *message);


#endif
