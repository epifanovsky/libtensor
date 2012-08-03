#ifndef LIBTENSOR_BTOD_CHOLESKY_H
#define LIBTENSOR_BTOD_CHOLESKY_H

#include "../core/block_tensor_i.h"

#include <libtensor/libtensor.h>


namespace libtensor {
/** \brief Computes a Cholesky decomposition of square matrix using LAPACK routine

    \ingroup libtensor_btod
 **/
class btod_cholesky {
public:
    btod_cholesky(block_tensor_i<2, double> &bta, double tol = 1e-4);
    //!< bta - input symmetric matrix
    virtual ~btod_cholesky();
    void decompose();
    //!<perform cholesky decomposition and save data to buffer
    int get_rank(){return m_rank;}; 
    virtual void perform(block_tensor_i<2, double> &btb);
    //!< put the data from buffer to output btensor, btb - output btensor
private:
    block_tensor_i<2, double> &m_bta; //!< Input block %tensor
    dense_tensor <2, double, std_allocator<double> > * pta;//!< buffer
    double m_tol; //!< tolerance for decomsposition
    int m_rank; //!< rank of decomposed matrix
};

}

#endif // LIBTENSOR_BTOD_CHOLESKY_H
