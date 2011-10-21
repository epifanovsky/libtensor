#ifndef LIBTENSOR_TOD_COPY_H
#define LIBTENSOR_TOD_COPY_H

#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "loop_list_add.h"
#include "loop_list_copy.h"
#include "tod_additive.h"
#include "bad_dimensions.h"

namespace libtensor {

/**	\brief Makes a copy of a %tensor, scales or permutes %tensor elements
        if necessary
    \tparam N Tensor order.

    This operation makes a scaled and permuted copy of a %tensor.
    The result can replace or be added to the output %tensor.


    <b>Examples</b>

    Plain copy:
    \code
    tensor_i<2, double> &t1(...), &t2(...);
    tod_copy<2> cp(t1);
    cp.perform(t2); // Copies the elements of t1 to t2
    \endcode

    Scaled copy:
    \code
    tensor_i<2, double> &t1(...), &t2(...);
    tod_copy<2> cp(t1, 0.5);
    cp.perform(t2); // Copies the elements of t1 multiplied by 0.5 to t2
    \endcode

    Permuted copy:
    \code
    tensor_i<2, double> &t1(...), &t2(...);
    permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
    tod_copy<2> cp(t1, perm);
    cp.perform(t2); // Copies transposed t1 to t2
    \endcode

    Permuted and scaled copy:
    \code
    tensor_i<2, double> &t1(...), &t2(...);
    permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
    tod_copy<2> cp(t1, perm, 0.5);
    cp.perform(t2); // Copies transposed t1 scaled by 0.5 to t2
    \endcode

    \ingroup libtensor_tod
 **/
template<size_t N>
class tod_copy: public loop_list_add,
    public loop_list_copy,
    public tod_additive<N> ,
    public timings<tod_copy<N> > {

public:
    static const char *k_clazz; //!< Class name

private:
    tensor_i<N,double> &m_ta; //!< Source %tensor
    permutation<N> m_perm; //!< Permutation of elements
    double m_c; //!< Scaling coefficient
    dimensions<N> m_dimsb; //!< Dimensions of output %tensor

public:
    //!	\name Construction and destruction
    //@{

    /**	\brief Prepares the copy operation
        \param ta Source %tensor.
        \param c Coefficient.
     **/
    tod_copy(tensor_i<N,double> &ta, double c = 1.0);

    /**	\brief Prepares the permute & copy operation
        \param ta Source %tensor.
        \param p Permutation of %tensor elements.
        \param c Coefficient.
     **/
    tod_copy(tensor_i<N,double> &ta, const permutation<N> &p, double c = 1.0);

    /**	\brief Virtual destructor
     **/
    virtual ~tod_copy() {
    }

    //@}


    //!	\name Implementation of libtensor::tod_additive<N>
    //@{

    virtual void prefetch();

    virtual void perform(cpu_pool &cpus, bool zero, double c,
        tensor_i<N, double> &tb);

    //@}

private:
    /**	\brief Creates the dimensions of the output using an input
            %tensor and a permutation of indexes
     **/
    static dimensions<N> mk_dimsb(tensor_i<N,double> &ta,
        const permutation<N> &perm);

    template<typename Base>
    void do_perform(cpu_pool &cpus, double c, tensor_i<N,double> &t);

    template<typename Base>
    void build_loop(typename Base::list_t &loop, const dimensions<N> &dimsa,
        const permutation<N> &perma, const dimensions<N> &dimsb);

};


} // namespace libtensor


#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class tod_copy<1>;
    extern template class tod_copy<2>;
    extern template class tod_copy<3>;
    extern template class tod_copy<4>;
    extern template class tod_copy<5>;
    extern template class tod_copy<6>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "tod_copy_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_TOD_COPY_H
