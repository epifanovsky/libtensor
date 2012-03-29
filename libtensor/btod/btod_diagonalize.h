#ifndef LIBTENSOR_BTOD_DIAGONALIZE_H
#define LIBTENSOR_BTOD_DIAGONALIZE_H

#include "../core/block_tensor_i.h"

namespace libtensor {

/** \brief Calculates the eigenvalues and eigenvectors of the symmetric
 *     tridiagonal matrix using QR iterations. The matrix must be in the
 *  tridiagonal form (see btod_tridiagonalize for details how to get it)

    \ingroup libtensor_btod
 **/

class btod_diagonalize
{
public:
    btod_diagonalize(block_tensor_i<2, double> &bta,block_tensor_i<2, double> &S,
            double tol = 1e-10,int maxiter = 10000);
    //!< bta - input tridiagonal matrix,S - matrix of transformation between
    //!< symmetric and tridiagonal matrix. Please plug unitary matrix if the
    //!< the matrix was in the tridiagonal form initially
    virtual void perform(block_tensor_i<2, double> &btb,
    block_tensor_i <2, double> &eigvector,
    block_tensor_i <1, double> &eigvalue);
    //!< btb - output diagonal matrix, eigvector - matrix which columns are
    //!< eigenvectors, eigvalue - vector containing eigenvalues
    virtual void print(block_tensor_i<2, double> &btb,
    block_tensor_i <2, double> &eigvector,block_tensor_i <1, double> &eigvalue);
    //!< prints the diagonal matrix, eigenvalues and matrix of eigenvectors
    virtual void check(block_tensor_i <2, double> &bta ,
            block_tensor_i <2, double> &eigvector,
    block_tensor_i <1, double> &eigvalue, double tol = 0);
    //!< checks if the eigenvalues and eigenvectors are obtained correctly
    virtual void sort(block_tensor_i <1, double> &eigvalue,size_t *map);
    //!< Map - mapping of the eigenvalues in decreasing order
    double get_eigenvalue(block_tensor_i <1, double> &eigvalue,size_t k);
    //!< Exctracts eigenvalue number k
    virtual void get_eigenvector(block_tensor_i <2, double> &eigvector,size_t k,
            block_tensor_i <1, double> &output);
    //!< Exctracts eigenvector number k
    double get_tol(){return m_tol;};
    //!< Show the tolerance
    int get_maxiter(){return m_maxiter;};
    //!< Show the number of max iterations
    int get_iter(){return m_iter;};
    //!< Show the number of iterations have been done
    bool get_doneiter(){return doneiter;};
    //!< Check if iterations are converged
    bool get_checked(){return checked;};
    //!< Returns true if the eigenvectors and eigenvalues have been checked
private:
    block_tensor_i<2, double> &m_bta; //!< Input block %tensor
    block_tensor_i<2, double> &m_S;//!< Inputmatrix of transformation
    double m_tol;//!< Tolerance
    int m_maxiter;//!< Maximum number of iterations
    int m_iter;//!< Number of iterations have been done
    bool doneiter;//!< iterations are converged?
    bool checked; //!< Eigenvalues and eigenvectors have been checked?
};

}

#endif // LIBTENSOR_BTOD_DIAGONALIZE_H
