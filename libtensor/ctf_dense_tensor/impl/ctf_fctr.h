#ifndef LIBTENSOR_CTF_FCTR_H
#define LIBTENSOR_CTF_FCTR_H

namespace libtensor {


/** \brief Element-wise division (c += alpha * a / b)

    \ingroup libtensor_ctf_dense_tensor
 **/
void ctf_fctr_ddiv(double alpha, double a, double b, double &c);


/** \brief Element-wise multiplication (b = alpha * b * a)

    \ingroup libtensor_ctf_dense_tensor
 **/
void ctf_fsum_dmul(double alpha, double a, double &b);


/** \brief Element-wise multiplication with addition (b = b + alpha * b * a)

    \ingroup libtensor_ctf_dense_tensor
 **/
void ctf_fsum_dmul_add(double alpha, double a, double &b);


/** \brief Element-wise division (b = alpha * b / a)

    \ingroup libtensor_ctf_dense_tensor
 **/
void ctf_fsum_ddiv(double alpha, double a, double &b);


/** \brief Element-wise division with addition (b = b + alpha * b / a)

    \ingroup libtensor_ctf_dense_tensor
 **/
void ctf_fsum_ddiv_add(double alpha, double a, double &b);


} // namespace libtensor

#endif // LIBTENSOR_CTF_FCTR_H

