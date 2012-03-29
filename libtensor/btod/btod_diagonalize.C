#include <iostream>
#include "../core/allocator.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/block_tensor.h"
#include "btod_add.h"
#include "btod_contract2.h"
#include "btod_copy.h"
#include "btod_diag.h"
#include "btod_diagonalize.h"
#include "btod_extract.h"
#include "btod_scale.h"
#include "btod_set_diag.h"
#include "btod_set_elem.h"
#include "btod_set.h"

namespace libtensor
{

btod_diagonalize::btod_diagonalize(block_tensor_i<2, double> &bta,
        block_tensor_i<2, double> &S,double tol,int maxiter) :
m_bta(bta),m_S(S),m_tol(tol),m_maxiter(maxiter),m_iter(0), doneiter(0),
checked(0){

}

void btod_diagonalize::perform(block_tensor_i<2, double> &btb,
        block_tensor_i <2, double> &eigvector,
        block_tensor_i <1, double> &eigvalue)
    {

    typedef std_allocator<double> allocator_t;

    btod_set_diag<2> (1).perform(eigvector);

    block_tensor<2, double, allocator_t> Q(btb.get_bis());//orthogonal matrix
    block_tensor<2, double, allocator_t> P(btb.get_bis());//rotation matrix
    block_tensor<2, double, allocator_t> Ptransp(btb.get_bis());//transposed
                                                                //rot matrix
    block_tensor<2, double, allocator_t> temp(btb.get_bis());// temporary matrix

    block_tensor_ctrl<2, double> cab(btb);

    //copy the input tensor
    btod_copy<2>(m_bta).perform(btb);

    const dimensions<2> *dimsa;
    dimsa = &(m_bta.get_bis().get_dims());
    size_t size = (*dimsa).get_dim(1);

    //check if the process has been converged
    index<2> idxc;
    idxc[0] = 0;
    idxc[1] = 0;
    size_t posc0=0;
    size_t posc1=1;

    doneiter = 1;

    for(size_t j=0;j<size-1;j++)
    {
        if(btb.get_bis().get_block_dims(idxc).get_dim(0) - 1 < posc0)
        {
            posc0 = 0;
            idxc[0]++;
        }

        if(btb.get_bis().get_block_dims(idxc).get_dim(1) - 1 < posc1)
        {
            posc1 = 0;
            idxc[1]++;
        }

        double elem;//x i

        {
        dense_tensor_ctrl<2, double> cbtbs(cab.req_block(idxc));
        const double *pas = cbtbs.req_const_dataptr();


        elem = *(pas + posc1 + posc0 * btb.get_bis().get_block_dims(idxc).
                get_dim(1));

        cbtbs.ret_const_dataptr(pas);
        }
        cab.ret_block(idxc);

        if(elem > m_tol)
        {
            doneiter = 0;
        }
        posc0++;
        posc1++;
    }

    //QR iterations
    while(m_iter < m_maxiter && doneiter==0)
    {

        btod_set<2>(0).perform(Q);
        btod_set_diag<2> (1).perform(Q);

        index<2> idx;

        size_t pos;
        size_t pos0;
        size_t pos1;

        bool done=0;

        //construct P matrices and from them update Q and R
        for(size_t i = 0; i < size - 1;i++)
        {

        btod_set<2>(0).perform(P);
        btod_set_diag<2> (1).perform(P);

        idx[0]=0;idx[1]=0;
        pos=i;

        done=0;

        //get ii element

        while(!done)
        {
            if(btb.get_bis().get_block_dims(idx).get_dim(0) - 1 < pos)
            {
                pos -= btb.get_bis().get_block_dims(idx).get_dim(0);
                idx[0]++;
                idx[1]++;
            }
            else
            {
                done = 1;
            }
        }


        double x;//x i

        {
        dense_tensor_ctrl<2, double> cbtb(cab.req_block(idx));
        const double *pa = cbtb.req_const_dataptr();

        x = *(pa + pos + pos * btb.get_bis().get_block_dims(idx).get_dim(1));

        cbtb.ret_const_dataptr(pa);
        }
        cab.ret_block(idx);

        double b;

        //get ii+1 element

        if(pos == btb.get_bis().get_block_dims(idx).get_dim(0) - 1)
        {
            idx[0]++;
            pos0=0;
            pos1=btb.get_bis().get_block_dims(idx).get_dim(1) - 1;

            {
            dense_tensor_ctrl<2, double> cbtb2(cab.req_block(idx));
            const double *pa2 = cbtb2.req_const_dataptr();

            b = *(pa2 + pos1 + pos0 * btb.get_bis().get_block_dims(idx).
                    get_dim(1));

            cbtb2.ret_const_dataptr(pa2);
            }
            cab.ret_block(idx);

        }
        else
        {
            pos0=pos+1;
            pos1=pos;


            {
            dense_tensor_ctrl<2, double> cbtb2(cab.req_block(idx));
            const double *pa2 = cbtb2.req_const_dataptr();

            b = *(pa2 + pos1 + pos0 * btb.get_bis().get_block_dims(idx).
                    get_dim(1));

            cbtb2.ret_const_dataptr(pa2);
            }

            cab.ret_block(idx);
        }

        //get sinus and cosinus
        double c = x/sqrt(b*b+x*x);
        double s = b/sqrt(b*b+x*x);

        index<2> idxibl;
        index<2> idx1;
        size_t prevsize;
        idx[0]=0;idx[1]=0;
        pos = i;
        done = 0;

        //set ii element
        while(!done)
        {
            if(P.get_bis().get_block_dims(idx).get_dim(0) - 1 < pos)
            {
                pos -= P.get_bis().get_block_dims(idx).get_dim(0);
                idx[0]++;
                idx[1]++;
            }
            else
            {
                done = 1;
            }
        }
        prevsize=P.get_bis().get_block_dims(idx).get_dim(0)-1;
        idx1[0]=idx[0];
        idx1[1]=idx[1];
        idxibl[0]=pos;
        idxibl[1]=pos;

        btod_set_elem<2>().perform(P,idx,idxibl,c);

        if(pos==prevsize)
        {
            //set i+1 i+1
            idx[0]=idx1[0]+1;
            idx[1]=idx1[1]+1;
            idxibl[0]=0;
            idxibl[1]=0;
            btod_set_elem<2>().perform(P,idx,idxibl,c);
            //set i i+1 element
            idx[0]=idx1[0];
            idx[1]=idx1[1]+1;
            idxibl[0]=P.get_bis().get_block_dims(idx).get_dim(0)-1;
            idxibl[1]=0;
            btod_set_elem<2>().perform(P,idx,idxibl,s);
            //set i+1 i element
            idx[0]=idx1[0]+1;
            idx[1]=idx1[1];
            idxibl[0]=0;
            idxibl[1]=P.get_bis().get_block_dims(idx).get_dim(1)-1;
            btod_set_elem<2>().perform(P,idx,idxibl,-s);
        }
        else
        {
            //set i+1 i+1
            idx[0]=idx1[0];
            idx[1]=idx1[1];
            idxibl[0]=pos+1;
            idxibl[1]=pos+1;
            btod_set_elem<2>().perform(P,idx,idxibl,c);
            //set i i+1 element
            idx[0]=idx1[0];
            idx[1]=idx1[1];
            idxibl[0]=pos;
            idxibl[1]=pos+1;
            btod_set_elem<2>().perform(P,idx,idxibl,s);
            //set i+1 i element
            idx[0]=idx1[0];
            idx[1]=idx1[1];
            idxibl[0]=pos+1;
            idxibl[1]=pos;
            btod_set_elem<2>().perform(P,idx,idxibl,-s);
        }

        //get transposed matrix
        permutation<2> perm;
        perm.permute(0, 1);
        btod_copy<2>(P,perm).perform(Ptransp);

        //get Q matrix
        contraction2<1,1,1> contrq;
        contrq.contract(1,0);
        btod_contract2<1,1,1>(contrq,Q,Ptransp).perform(temp);
        btod_copy<2>(temp).perform(Q);

        //update matrix by rotations
        btod_contract2<1,1,1>(contrq,P,btb).perform(temp);
        btod_copy<2>(temp).perform(btb);
        }
        contraction2<1,1,1> contr;
        contr.contract(1,0);
        btod_contract2<1,1,1>(contr,btb,Q).perform(temp);

        btod_copy<2>(temp).perform(btb);

        //computation of eigenvectors
        btod_contract2<1,1,1>(contr,eigvector,Q).perform(temp);
        btod_copy<2>(temp).perform(eigvector);
        //check if the process has been converged

        doneiter = 1;
        idxc[0] = 0;
        idxc[1] = 0;
        posc0=0;
        posc1=1;
        for(size_t j=0;j<size-1;j++)
        {

            if(btb.get_bis().get_block_dims(idxc).get_dim(0) - 1 < posc0)
            {
                posc0 = 0;
                idxc[0]++;
            }

            if(btb.get_bis().get_block_dims(idxc).get_dim(1) - 1 < posc1)
            {
                posc1 = 0;
                idxc[1]++;
            }

            double elem;//x i
            {
            dense_tensor_ctrl<2, double> cbtbs(cab.req_block(idxc));
            const double *pas = cbtbs.req_const_dataptr();

            elem = *(pas + posc1 + posc0 * btb.get_bis().get_block_dims(idxc).
                    get_dim(1));

            cbtbs.ret_const_dataptr(pas);
            }
            cab.ret_block(idxc);

            if(elem<0)
            {
                elem = - elem;
            }

            if(elem > m_tol)
            {
                doneiter = 0;
            }

            posc0++;
            posc1++;

        }
        m_iter++;
    }//end of iterations

    //get eigenvalues
    mask<2> msk;
    msk[0] = true; msk[1] = true;
    btod_diag<2, 2>(btb, msk).perform(eigvalue);
    //get matrix of eigenvectors
    contraction2<1,1,1> contr;
    contr.contract(1,0);
    btod_contract2<1,1,1>(contr,m_S,eigvector).perform(temp);
    btod_copy<2>(temp).perform(eigvector);
    }



void btod_diagonalize::print(block_tensor_i<2, double> &btb,
block_tensor_i <2, double> &eigvector,block_tensor_i <1, double> &eigvalue)
{
    block_tensor_ctrl<2, double> ctrla(m_bta);
    block_tensor_ctrl<2, double> ctrlb(btb);
    block_tensor_ctrl<2, double> ctrlevec(eigvector);
    block_tensor_ctrl<1, double> ctrleigval(eigvalue);

    size_t size = m_bta.get_bis().get_dims().get_dim(1);
    std::cout<<"The tridiagonal matrix A is:"<<std::endl;
    std::cout<<std::endl;
    index<2> idxi;
    idxi[0] = 0;
    int pos0 = 0;

    for(size_t i =0;i < size;i++)
    {
        if(m_bta.get_bis().get_block_dims(idxi).get_dim(0) - 1 < pos0)
        {
            pos0 = 0;
            idxi[0]++;
        }

        idxi[1] = 0;
        int pos1 = 0;

        for(size_t j =0;j < size;j++)
        {
            if(m_bta.get_bis().get_block_dims(idxi).get_dim(1) - 1 < pos1)
            {

                pos1 = 0;
                idxi[1]++;
            }

            if(ctrla.req_is_zero_block(idxi)==false)
            {
                {
            dense_tensor_ctrl<2, double> catrl(ctrla.req_block(idxi));
            const double *pa = catrl.req_const_dataptr();
            std::cout<<*(pa + pos0 * m_bta.get_bis().get_block_dims(idxi).
                    get_dim(1) + pos1)<<" ";
            catrl.ret_const_dataptr(pa);
                }
            ctrla.ret_block(idxi);
            }
            else
            {
            std::cout<<"'"<<" ";
            }
            pos1++;
        }

        std::cout<<std::endl;
        pos0++;
    }

    std::cout<<std::endl;



    if(doneiter==0)
    {
        std::cout<<std::endl;
        std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                <<std::endl;
        std::cout<<"ERROR! THE NUMBER OF ITERATIONS "<<m_maxiter<<
                " HAS BEEN EXHAUSTED.";
        std::cout<<"NO CONVERGENCE FOR TOLERANCE "<<m_tol<<
                " HAS BEEN ACHIEVED"<<std::endl;
        std::cout<<
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                <<std::endl;
        std::cout<<std::endl;
    }

    idxi[0] = 0;
    pos0 = 0;

    std::cout<<"The diagonal matrix for matrix A is:"<<std::endl;
    std::cout<<std::endl;

    for(size_t i =0;i < size;i++)
    {
        if(btb.get_bis().get_block_dims(idxi).get_dim(0) - 1 < pos0)
        {
            pos0 -= btb.get_bis().get_block_dims(idxi).get_dim(0);
            idxi[0]++;
        }

        idxi[1] = 0;
        int pos1 = 0;

        for(size_t j = 0; j < size; j++)
        {
            if(btb.get_bis().get_block_dims(idxi).get_dim(1) - 1 < pos1)
            {
                pos1 -= btb.get_bis().get_block_dims(idxi).get_dim(1);
                idxi[1]++;
            }

            if(ctrlb.req_is_zero_block(idxi)==false)
            {
                {
                    dense_tensor_ctrl<2, double> catrl(ctrlb.req_block(idxi));
                    const double *pa = catrl.req_const_dataptr();
                    std::cout<<*(pa + pos0 * btb.get_bis().get_block_dims(idxi).
                            get_dim(1) + pos1)<<" ";
                    catrl.ret_const_dataptr(pa);
                }
                ctrlb.ret_block(idxi);
            }
            else
            {
                std::cout<<"'"<<" ";
            }
            pos1++;
        }

        std::cout<<std::endl;
        pos0++;
    }


    idxi[0] = 0;
    pos0 = 0;

    std::cout<<std::endl;
    std::cout<<"The matrix of eigenvectors for matrix A is:"<<std::endl;
    std::cout<<std::endl;

    for(size_t i =0;i < size;i++)
    {
        if(eigvector.get_bis().get_block_dims(idxi).get_dim(0) - 1 < pos0)
        {
            pos0 -= eigvector.get_bis().get_block_dims(idxi).get_dim(0);
            idxi[0]++;
        }

        idxi[1] = 0;
        int pos1 = 0;

        for(size_t j =0;j < size;j++)
        {
            if(eigvector.get_bis().get_block_dims(idxi).get_dim(1) - 1 < pos1)
            {
                pos1 -= eigvector.get_bis().get_block_dims(idxi).get_dim(1);
                idxi[1]++;
            }

            if(ctrlevec.req_is_zero_block(idxi)==false)
            {
                {
            dense_tensor_ctrl<2, double> catrl(ctrlevec.req_block(idxi));
            const double *pa = catrl.req_const_dataptr();
            std::cout<<*(pa + pos0 * eigvector.get_bis().get_block_dims(idxi).
                    get_dim(1) + pos1)<<" ";
            catrl.ret_const_dataptr(pa);
                }
            ctrlevec.ret_block(idxi);
            }
            else
            {
                std::cout<<"'"<<" ";
            }
            pos1++;
        }

        std::cout<<std::endl;
        pos0++;
    }



        std::cout<<std::endl;
            std::cout<<"Eigenvalues are "<<std::endl;
            std::cout<<std::endl;

            for(size_t k =0; k < eigvalue.get_bis().get_dims().get_dim(0);k++)
            {
                get_eigenvalue(eigvalue,k);
            }
            std::cout<<std::endl;

    std::cout<<"The number of iterations is "<< m_iter <<std::endl;
    std::cout<<std::endl;
    std::cout<<"Checking ontained vectors and eigenvalues:"<<std::endl;
    if(checked==1)
    {
        std::cout<<"SUCCESSFULLY!"<<std::endl;
    }
    else
    {
        std::cout<<"FAIL!"<<std::endl;
    }
    std::cout<<std::endl;
    std::cout<<"================================================"<<std::endl;
    std::cout<<"------------------------------------------------"<<std::endl;
    std::cout<<"================================================"<<std::endl;
    std::cout<<std::endl;


}

void btod_diagonalize::check(block_tensor_i <2, double> &bta ,
        block_tensor_i <2, double> &eigvector,
        block_tensor_i <1, double> &eigvalue,double tol)
{
    if(tol==0.0)
    {
        tol = m_tol;
    }
    typedef std_allocator<double> allocator_t;
    block_tensor_ctrl<1, double> ctrleigval(eigvalue);

    block_tensor<1, double, allocator_t> zero(eigvalue.get_bis());
    block_tensor<1, double, allocator_t> vector(eigvalue.get_bis());
    block_tensor<1, double, allocator_t> temp(eigvalue.get_bis());

    block_tensor_ctrl<1, double> ctrlzero(zero);
    checked = 1;

    index<1> idxv;
    idxv[0]=0;
    size_t pos = 0;

        for(size_t k =0; k < eigvalue.get_bis().get_dims().get_dim(0);k++)
        {

            //get eigenvalue
            double lamda;
            lamda = get_eigenvalue(eigvalue,k);

            //get eigen vector
            get_eigenvector(eigvector,k,vector);

            contraction2<1,0,1> contr;
            contr.contract(1,0);
            btod_contract2<1,0,1>(contr,bta,vector).perform(temp);

            btod_scale<1>(vector,lamda).perform();

            btod_add<1> op(temp);
            op.add_op(vector,-1);
            op.perform(zero);

            //Check if vector zero is zero
            index<1> idxv2;
            idxv2[0]=0;
            size_t pos2 = 0;

                for(size_t k2 =0; k2 < zero.get_bis().get_dims().get_dim(0);
                        k2++)
                {
                    if(zero.get_bis().get_block_dims(idxv2).get_dim(0)-1 <pos2)
                    {
                        idxv2[0]++;
                        pos2=0;
                    }
                    if(ctrlzero.req_is_zero_block(idxv2) == false)
                    {
                        double value;

                        {
                    dense_tensor_ctrl<1, double> cavp2(ctrlzero.req_block(idxv2));

                    const double *pav2 = cavp2.req_const_dataptr();

                    value = *(pav2 + pos2);
                    if(value < 0)
                    {
                        value = -value;
                    }

                    if(value>tol)
                    {
//                      std::cout<<"Bad value is"<<*(pav2 + pos2)<<std::endl;
                        checked = 0;
                    }

                cavp2.ret_const_dataptr(pav2);
                        }
                ctrlzero.ret_block(idxv2);
                    }
                pos2++;
                }

        pos++;
        }
}

void btod_diagonalize::sort(block_tensor_i <1, double> &eigvalue,size_t *map)
        {
    block_tensor_ctrl<1, double> ctrleigvalue(eigvalue);
            for(size_t i = 0; i < eigvalue.get_bis().get_dims().get_dim(0);i++)
            {
                map[i] = i;
            }

            double x1;
            double x2;

            //Bubble sorts the eigenvalues in decreasing order
            int j=0;
            do
            {
                index<1> idxv2;
                idxv2[0]=0;
                size_t pos2 = 0;
                j=0;
                for (size_t i=0;i<eigvalue.get_bis().get_dims().get_dim(0)-1;
                        i++)
                {
                            pos2=map[i];
                            idxv2[0]=0;
                            bool done = false;
                            while(done==false)
                            {
                            if(eigvalue.get_bis().get_block_dims(idxv2).
                                    get_dim(0)-1 <pos2)
                            {
                                pos2-=eigvalue.get_bis().get_block_dims(idxv2).
                                        get_dim(0);
                                idxv2[0]++;
                            }
                            else
                            {
                                done = true;
                            }
                            }
                            if(ctrleigvalue.req_is_zero_block(idxv2) == false)
                            {
                                {
                            dense_tensor_ctrl<1, double> cavp2(ctrleigvalue.
                                    req_block(idxv2));

                            const double *pav2 = cavp2.req_const_dataptr();

                            x1=*(pav2 + pos2);

                        cavp2.ret_const_dataptr(pav2);
                                }
                        ctrleigvalue.ret_block(idxv2);
                            }
                            pos2=map[i+1];
                            idxv2[0]=0;
                            done = false;
                            while(done==false)
                            {
                            if(eigvalue.get_bis().get_block_dims(idxv2).
                                    get_dim(0)-1 <pos2)
                                {
                                pos2-=eigvalue.get_bis().get_block_dims(idxv2).
                                        get_dim(0);
                                    idxv2[0]++;
                                }
                            else
                            {
                                done=true;
                            }
                            }
                                if(ctrleigvalue.req_is_zero_block(idxv2) ==
                                        false)
                                {
                                    {
                                dense_tensor_ctrl<1, double> cavp2(ctrleigvalue.
                                        req_block(idxv2));

                                const double *pav2 = cavp2.req_const_dataptr();

                                x2=*(pav2 + pos2);

                            cavp2.ret_const_dataptr(pav2);
                                }
                            ctrleigvalue.ret_block(idxv2);
                                }

                    if(x1*x1<x2*x2)
                    {
                        j++;
                        size_t j1;
                        size_t j2;
                        j1 = map[i];
                        j2 = map[i+1];
                        map[i] = j2;
                        map[i+1] = j1;
                    }
                }
            } while(j!=0);
        }

double btod_diagonalize::get_eigenvalue(block_tensor_i <1, double> &eigvalue,
        size_t k)
{
    block_tensor_ctrl<1, double> ctrleigval(eigvalue);
    index<1> idxv;
    double answer;
    bool done;
    done = false;
    idxv[0]=0;
    size_t pos = k;
    while(done==false)
                {
                    if(eigvalue.get_bis().get_block_dims(idxv).get_dim(0)-1 <
                            pos)
                    {
                        pos-=eigvalue.get_bis().get_block_dims(idxv).get_dim(0);
                        idxv[0]++;
                    }
                    else
                    {
                        done = true;
                    }
                }

                    if(ctrleigval.req_is_zero_block(idxv) == false)
                    {
                        {
                    dense_tensor_ctrl<1, double> cavp(ctrleigval.req_block(idxv));

                    const double *pav = cavp.req_const_dataptr();

                    answer=*(pav + pos);
                cavp.ret_const_dataptr(pav);
                        }
                ctrleigval.ret_block(idxv);
                    }
                    else
                    {
                        answer = 0;
                    }
return answer;
}

void btod_diagonalize::get_eigenvector(block_tensor_i <2, double> &eigvector,size_t k,
        block_tensor_i <1, double> &output)
        {
        bool done;
        index<2> idx;
        size_t pos;
        idx[0]=0;idx[1]=0;
        pos=k;
        done=0;

        while(!done)
        {
            if(eigvector.get_bis().get_block_dims(idx).get_dim(1) - 1 < pos)
            {
                pos -= eigvector.get_bis().get_block_dims(idx).get_dim(1);
                idx[1]++;
            }
            else
            {
                done = 1;
            }
        }

        mask<2> msk;
        msk[0] = true; msk[1] = false;
        index<2> idxibl;
        idxibl[0] = 0; idxibl[1] = pos;

        btod_extract<2, 1>(eigvector, msk, idx, idxibl).perform(output);
        }

}//namespace libtensor

