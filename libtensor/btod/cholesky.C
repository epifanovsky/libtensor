#include <sstream>
#include "../core/allocator.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/block_tensor.h"
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod/btod_add.h>
#include <libtensor/block_tensor/btod/btod_contract2.h>
#include <libtensor/block_tensor/btod/btod_copy.h>
#include <libtensor/block_tensor/btod/btod_diag.h>
#include "cholesky.h"
#include <libtensor/block_tensor/btod/btod_extract.h>
#include "btod_scale.h"
#include "btod_set_diag.h"
#include "btod_set_elem.h"
#include <libtensor/block_tensor/btod/btod_set.h>
#include "btod_select.h"

#include "btod_print.h"

//#define PRINT


namespace libtensor
{

cholesky::cholesky(block_tensor_i<2, double> &bta, double tol) :
m_bta(bta) ,m_tol(tol), m_rank(0)
, pta(new block_tensor<2, double, std_allocator <double> >(bta.get_bis()))
{

}

cholesky::~cholesky(){
        delete pta;
        pta = NULL;
}


void cholesky::decompose()
{
    
    typedef std_allocator<double> allocator_t;
    block_tensor_i<2, double> &buff(*pta);// buffer
    block_tensor<2, double, std_allocator <double> > D(buff.get_bis());// Residual matrix

    btod_copy<2>(m_bta).perform(buff); // now buffer has imput matrix
    btod_copy<2>(buff).perform(D); // now D has input matrix
    
    // find Cholesky
    size_t n = buff.get_bis().get_dims().get_dim(0);//size of the matrix

    index<2> idx;
        idx[1] = 0;

    size_t pos1 = 0;

    for(size_t i1 = 0; i1 < n ; i1++)
    {
    // first run over index 1 (columns of the matrices)
        if(buff.get_bis().get_block_dims(idx).get_dim(1) - 1 < pos1)
        {
                pos1 -= buff.get_bis().get_block_dims(idx).get_dim(1);
                idx[1]++;
        }

        idx[0] = 0;
        size_t pos0 = 0;

        index<2> idxibl; // index inside block
        idxibl[0] = 0; idxibl[1] = 0;

        index<2> idxbl; // index of the block
        idxbl[0] = 0; idxbl[1] = 0; 

    double diag = 0;
    diag = sort_diag(D, idxbl, idxibl);
    // extracted the maximum diagonal element

    if(diag < m_tol) break;

    m_rank++; // increase the rank

    // mask for column extraction

    mask<2> msk;
    msk[0] = true; msk[1] = false;
    
    btod_extract<2, 1> ex(D, msk, idxbl, idxibl);

    block_tensor<1, double, std_allocator <double> > column(ex.get_bis());

    ex.perform(column);
    // now column has a bt with column

    // scale the element

    dense_tensor <1 , double, std_allocator <double> > columnt(column.get_bis().get_dims());

    btod_scale<1>(column,1/sqrt(diag)).perform();

        tod_btconv<1>(column).perform(columnt); // now columnt has cholesky vector

    // update residual D^{j} = D^{j-1} - L^{j}*L^{j}'

    contraction2<1,1,0> contr;
        btod_contract2<1,1,0> opcntr(contr,column,column);

    opcntr.perform(D, -1.0);

    
    //save the vector to the buffer

    for(size_t i0 = 0 ; i0 < n; i0++)
    {
    // after that run over index 0 (rows of the matrix)

                if(buff.get_bis().get_block_dims(idx).get_dim(0) - 1 < pos0)
                {
                        pos0 -= buff.get_bis().get_block_dims(idx).get_dim(0);
                        idx[0]++;
                }

    block_tensor_ctrl<2, double> ctrl(buff);

        dense_tensor_i<2, double> &t = ctrl.req_block(idx);
        dense_tensor_ctrl<2, double> ct(t);
           double *p = ct.req_dataptr();

    size_t np = buff.get_bis().get_block_dims(idx).get_dim(1);

    dense_tensor_ctrl<1, double> ct2(columnt);
        const double *px = ct2.req_const_dataptr();
    
    *(p + pos0 * np + pos1) = *(px + i0) ;
    // put the Cholesky vector to the buffer

    ct.ret_dataptr(p);
        ctrl.ret_block(idx);

    ct2.ret_const_dataptr(px);

    pos0++;
    }

    pos1++;
    
    std::stringstream os;
        os<<std::endl;
    #ifdef PRINT
    std::cout<<"The matrix for the rank = "<<m_rank<<std::endl;
    btod_print<2>(os).perform(buff);
    std::cout<<os.str()<<std::endl;
    #endif

    }


}

double cholesky::sort_diag(block_tensor_i<2, double> &D, index<2> &idxbl, index<2> &idxibl)
{
    typedef std_allocator<double> allocator_t;

    // extract diagonal

        size_t nd = D.get_bis().get_dims().get_dim(0);

        mask<2> msk;
        msk[0] = true; msk[1] = true;
    // mask to extract diagonal
        btod_diag<2, 2> d(D, msk);
        block_tensor<1, double, std_allocator <double> > diagbt(d.get_bis());
        d.perform(diagbt);

    // now diagbt has diagonal

    // find the highest diag element

    #ifdef PRINT
        std::stringstream os;
        os<<std::endl;    
    os<<"Diagonal is "<<std::endl;
    btod_print<1>(os).perform(diagbt);

        #endif

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // here should change it to Zhenya's procedure

        double maxdiag = 0;
        size_t i1 = 0; // number of the maximum element
        size_t pos = 0;

     btod_select<1, compare4absmax>::list_t lst1;
    btod_select<1, compare4absmax>(diagbt).perform(lst1, 1);

           if(lst1.begin() != lst1.end()) {
                        maxdiag = lst1.begin()->value;
            index<1> bidx;
                bidx = lst1.begin()->bidx;
            index<1> idx;
                  idx = lst1.begin()->idx;
            idxbl[0] = 0; idxbl[1] = bidx[0];
            idxibl[0] = 0; idxibl[1] = idx[0];
            // here it might be a bug if
            // bis of diag not equal to the
            // bis of the columns of the
            // matrix
        }

    return maxdiag;
}

void cholesky::perform(block_tensor_i<2, double> &btb)
{
    
    size_t n = btb.get_bis().get_dims().get_dim(0);
    size_t R = m_rank ;
    
    block_tensor_i<2, double> &buff(*pta);
    
    index<2> idxi;
    idxi[0] = 0;

        index<2> idxo;
        idxo[0] = 0;

    size_t posi0 = 0;
    size_t poso0 = 0;
    
    for(size_t i0 = 0; i0 < n;i0++)
    {

    if(buff.get_bis().get_block_dims(idxi).get_dim(0) - 1 < posi0)
    {
        posi0 -= buff.get_bis().get_block_dims(idxi).get_dim(0);
        idxi[0]++;
    }

        if(btb.get_bis().get_block_dims(idxo).get_dim(0) - 1 < poso0)
        {
                poso0 -= btb.get_bis().get_block_dims(idxo).get_dim(0);
                idxo[0]++;
        }
    
    idxi[1] = 0;
    idxo[1] = 0;
    size_t posi1 = 0;
    size_t poso1 = 0;

    for(size_t i1 = 0; i1 < R; i1++)
    {
        if(buff.get_bis().get_block_dims(idxi).get_dim(1) - 1 < posi1)
        {
            posi1 -= buff.get_bis().get_block_dims(idxi).get_dim(1);
            idxi[1]++;
        }
        
        if(btb.get_bis().get_block_dims(idxo).get_dim(1) - 1 < poso1)
                {
                        poso1 -= btb.get_bis().get_block_dims(idxo).get_dim(1);
                        idxo[1]++;
                }

        block_tensor_ctrl<2, double> ctrli(buff);
        block_tensor_ctrl<2, double> ctrlo(btb);

        if(ctrli.req_is_zero_block(idxi)==false)
        {
    
        
        dense_tensor_i<2, double> &ti = ctrli.req_block(idxi);
        dense_tensor_ctrl<2, double> cti(ti);
        const double *pi = cti.req_const_dataptr();        
        
        dense_tensor_i<2, double> &to = ctrlo.req_block(idxo);
        dense_tensor_ctrl<2, double> cto(to);
        double *po = cto.req_dataptr();

        size_t Rp = btb.get_bis().get_block_dims(idxo).get_dim(1); 
        size_t np = buff.get_bis().get_block_dims(idxi).get_dim(1);

        *(po +poso0 * Rp + poso1) = *(pi + posi0 * np + posi1);

        cti.ret_const_dataptr(pi);
        cto.ret_dataptr(po);
        pi=0;
        po=0;

        ctrli.ret_block(idxi);
        ctrlo.ret_block(idxo);
        
        }
        posi1++;
        poso1++;
    }
    
    posi0++;
    poso0++;
    }

}

}//namespace libtensor

