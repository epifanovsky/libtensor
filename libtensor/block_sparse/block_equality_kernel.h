#ifndef BLOCK_EQUALITY_KERNEL_H
#define BLOCK_EQUALITY_KERNEL_H

#include "block_kernel_i.h"


//TODO REMOVE
#include <iostream>

namespace libtensor {

template<typename T>
class block_equality_kernel : public block_kernel_i<0,2,T> {
public: 
    static const char *k_clazz; //!< Class name
private:
    bool m_run_once;
    bool m_equal;
public:
    block_equality_kernel() : m_run_once(false), m_equal(true) { };
    void operator()(sequence<0, T*>& output_ptrs, 
                    sequence<2, T*>& input_ptrs,
                    sequence<0, dim_list>& output_dims,
                    sequence<2, dim_list >& input_dims);

    bool equal() const throw(bad_parameter);

    //Copy constructor
    block_equality_kernel(const block_equality_kernel<T>& rhs) : m_run_once(rhs.m_run_once),m_equal(rhs.m_equal) { }; 

    //Returns a pointer to a copy of this object
    block_kernel_i<0,2,T>* clone() const { return (block_kernel_i<0,2,T>*) new block_equality_kernel<T>(*this); };
};

template<typename T>
const char *block_equality_kernel<T>::k_clazz = "block_equality_kernel<T>";

template<typename T>
void block_equality_kernel<T>::operator()(sequence<0, T*>& output_ptrs, 
                                          sequence<2, T*>& input_ptrs,
                                          sequence<0, dim_list>& output_dims,
                                          sequence<2, dim_list >& input_dims)
{

    dim_list& ib_0_dims = input_dims[0];
    dim_list& ib_1_dims = input_dims[1];

    T* ib_0_data_ptr = input_ptrs[0];
    T* ib_1_data_ptr = input_ptrs[1];

    if(ib_0_dims.size() !=  ib_1_dims.size())
    {
        throw bad_parameter(g_ns, k_clazz,"operator(...)",
                __FILE__, __LINE__, "both input blocks do not have the same dimensionality");
    }
    
    for(int i = 0; i < ib_0_dims.size(); ++i)
    {
        if(ib_0_dims[i] != ib_1_dims[i])
        {
            throw bad_parameter(g_ns, k_clazz, "operator(...)",
                    __FILE__, __LINE__, "input block dimensions do not match");
        }
    }

    size_t block_size = 1;
    for(int i = 0; i < ib_1_dims.size(); ++i)
    {
        block_size *= ib_1_dims[i];
    }

    m_run_once = true;
    for(size_t m = 0; m < block_size; ++m)
    {
        if(ib_0_data_ptr[m] != ib_1_data_ptr[m])
        {
            m_equal = false;
            break;
        }
    }
}

template<typename T>
bool block_equality_kernel<T>::equal() const
{
    if(!m_run_once)
    {
        throw bad_parameter(g_ns, k_clazz, "equal(...)",
                __FILE__, __LINE__, "no values have been compared yet!");
    }
    return m_equal;
}

} // namespace libtensor

#endif /* BLOCK_EQUALITY_KERNEL_H */
