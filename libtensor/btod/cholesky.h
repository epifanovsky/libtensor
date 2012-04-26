#ifndef LIBTENSOR_CHOLESKY_H
#define LIBTENSOR_CHOLESKY_H

#include "../core/block_tensor_i.h"

namespace libtensor {

/** \brief Computes Cholesky decomposition

    \ingroup libtensor_btod
 **/

class cholesky
{
public:
    cholesky(block_tensor_i<2, double> &bta, double tol = 1e-4);
    //!< bta - input matrix.
    ~cholesky();
    virtual void decompose();
    //!< perform decomposition
    virtual void perform(block_tensor_i<2, double> &btb);
    //!< btb - output matrix. put data to the output matrix
    double get_tol(){return m_tol;};
    //!< Show the tolerance
    int get_rank() {return m_rank;};
    //!< Get the rank of the decomposed matrix
private:
    block_tensor_i<2, double> &m_bta; //!< Input block %tensor
    double m_tol;//!< Tolerance
    int m_rank;
    block_tensor <2, double, std_allocator<double> > * pta;//!< buffer
    double sort_diag(block_tensor_i<2, double> &D, index<2> &idxbl, index<2> &idxibl);
};

}

#endif // LIBTENSOR_CHOLESKY_H
