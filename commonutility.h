#ifndef _COMMONUTILITY_H_
#define _COMMONUTILITY_H_

#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include <random>

#include "CSC.h"
#include "CSR.h"
#include "IO.h"

using namespace std;


template <class INDEXTYPE, class VALUETYPE>
void SetInputMatricesAsCSC(CSC<INDEXTYPE, VALUETYPE> &A_csc, string inputfile)
{
	string inputname;
    	A_csc.make_empty();
        inputname = inputfile;
#ifdef PRINTMSG
        cout << "Reading input matrices in text (ascii)... " << endl;
	cout << "Input File Directory:" << inputname << endl;
#endif
        ReadASCII(inputname, A_csc);
        stringstream ss1(inputname);
        string cur;
        
	vector<string> v1;
        while (getline(ss1, cur, '.')) {
            v1.push_back(cur);
        }
	inputname = v1[v1.size() - 1];
}

template <class INDEXTYPE, class VALUETYPE>
void SetInputMatricesAsCSR(CSR<INDEXTYPE, VALUETYPE> &A_csr, string inputfile)
{
    CSC<INDEXTYPE, VALUETYPE> A_csc;

    A_csr.make_empty();

    SetInputMatricesAsCSC(A_csc, inputfile);

    A_csr = *(new CSR<INDEXTYPE, VALUETYPE>(A_csc));
}

template <class INDEXTYPE, class VALUETYPE>
void printCSR(CSR<INDEXTYPE, VALUETYPE> &A_csr){
	cout << "Size of Row PTRS:" << A_csr.rows << ", Size of Col IDS:" << A_csr.nnz << endl;
	cout << "Rowptrs:";
        for(INDEXTYPE i = 0; i < A_csr.rows + 1; i++){
                cout << A_csr.rowptr[i] << " ";
        }
        cout << endl;
        cout << "Colids:";
        for(INDEXTYPE i = 0; i < A_csr.nnz + 1; i++){
                cout << A_csr.colids[i] << " ";
        }
        cout << endl;
        cout << "Values:";
        for(INDEXTYPE i = 0; i < A_csr.nnz + 1; i++){
                cout << A_csr.values[i] << " ";
        }
        cout << endl;
}

template <class INDEXTYPE, class VALUETYPE>
void printCSC(CSC<INDEXTYPE, VALUETYPE> &A_csc){
        cout << "Size of Col PTRS:" << A_csc.cols << ", Size of Row IDS:" << A_csc.nnz << endl;
        cout << "Coltrs:";
        for(INDEXTYPE i = 0; i < A_csc.cols + 1; i++){
                cout << A_csc.colptr[i] << " ";
        }
        cout << endl;
        cout << "Rowids:";
        for(INDEXTYPE i = 0; i < A_csc.nnz + 1; i++){
                cout << A_csc.rowids[i] << " ";
        }
        cout << endl;
        cout << "Values:";
        for(INDEXTYPE i = 0; i < A_csc.nnz + 1; i++){
                cout << A_csc.values[i] << " ";
        }
        cout << endl;
}
#endif
