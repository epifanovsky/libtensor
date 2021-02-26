#ifndef LIBTENSOR_BTO_DIAGONALIZE_H
#define LIBTENSOR_BTO_DIAGONALIZE_H

#include <libtensor/block_tensor/block_tensor_i.h>

namespace libtensor {

/** \brief Calculates the eigenvalues and eigenvectors of the symmetric
 *     tridiagonal matrix using QR iterations. The matrix must be in the
 *  tridiagonal form (see bto_tridiagonalize for details how to get it)

    \ingroup libtensor_btod
 **/
template <typename T>
class bto_diagonalize
{
public:
    bto_diagonalize(block_tensor_i<2, T> &bta,block_tensor_i<2, T> &S,
            T tol = 1e-10,int maxiter = 10000);
    //!< bta - input tridiagonal matrix,S - matrix of transformation between
    //!< symmetric and tridiagonal matrix. Please plug unitary matrix if the
    //!< the matrix was in the tridiagonal form initially
    virtual void perform(block_tensor_i<2, T> &btb,
    block_tensor_i <2, T> &eigvector,
    block_tensor_i <1, T> &eigvalue);
    //!< btb - output diagonal matrix, eigvector - matrix which columns are
    //!< eigenvectors, eigvalue - vector containing eigenvalues
    virtual void print(block_tensor_i<2, T> &btb,
    block_tensor_i <2, T> &eigvector,block_tensor_i <1, T> &eigvalue);
    //!< prints the diagonal matrix, eigenvalues and matrix of eigenvectors
    virtual void check(block_tensor_i <2, T> &bta ,
            block_tensor_i <2, T> &eigvector,
    block_tensor_i <1, T> &eigvalue, T tol = 0);
    //!< checks if the eigenvalues and eigenvectors are obtained correctly
    virtual void sort(block_tensor_i <1, T> &eigvalue,size_t *map);
    //!< Map - mapping of the eigenvalues in decreasing order
    T get_eigenvalue(block_tensor_i <1, T> &eigvalue,size_t k);
    //!< Exctracts eigenvalue number k
    virtual void get_eigenvector(block_tensor_i <2, T> &eigvector,size_t k,
            block_tensor_i <1, T> &output);
    //!< Extracts eigenvector number k
    T get_tol(){return m_tol;};
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
    block_tensor_i<2, T> &m_bta; //!< Input block %tensor
    block_tensor_i<2, T> &m_S;//!< Inputmatrix of transformation
    T m_tol;//!< Tolerance
    int m_maxiter;//!< Maximum number of iterations
    int m_iter;//!< Number of iterations have been done
    bool doneiter;//!< iterations are converged?
    bool checked; //!< Eigenvalues and eigenvectors have been checked?
};


}

#endif // LIBTENSOR_BTO_DIAGONALIZE_H
