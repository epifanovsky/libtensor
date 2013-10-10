#ifndef SPARSE_BTENSOR_H
#define SPARSE_BTENSOR_H

#include <sstream>
#include "sparse_bispace.h"
#include "block_loop.h"
#include "block_kernels.h"

//TODO: REMOVE
#include <iostream>

namespace libtensor {

template<size_t N,typename T = double>
class sparse_btensor {
public:
    static const char *k_clazz; //!< Class name
private:
    T* m_data;
    sparse_bispace<N> m_bispace;
public:
    /** \brief Constructs a sparse block tensor object and populates it with the entries from mem if specified
     **/
    sparse_btensor(const sparse_bispace<N>& the_bispace,T* mem = NULL,bool already_block_major = false);
    virtual ~sparse_btensor();

    /** \brief Return the sparse_bispace defining this tensor 
     **/
    sparse_bispace<N> get_bispace(); 

    /** \brief Compares the tensor to a chunk of memory stored by default in row major order
     **/
    bool operator==(const sparse_btensor<N,T>& rhs) const;

    /** \brief Returns a string representation of the tensor in row-major order 
     **/
    std::string str() const;
};

template<size_t N,typename T>
const char *sparse_btensor<N,T>::k_clazz = "sparse_btensor<N,T>";

template<size_t N,typename T>
sparse_btensor<N,T>::sparse_btensor(const sparse_bispace<N>& the_bispace,T* mem,bool already_block_major) : m_bispace(the_bispace)
{
    //Determine size
    size_t size = 1;
    for(size_t i = 0; i < N; ++i)
    {
        size *= the_bispace[i].get_dim();
    }

    //Alloc storage
    m_data = new T[size];

    if(mem != NULL)
    {
        if(already_block_major)
        {
            memcpy(m_data,mem,the_bispace.get_nnz()*sizeof(T));
        }
    }
}

template<size_t N,typename T>
sparse_btensor<N,T>::~sparse_btensor()
{
    delete [] m_data;
}

template<size_t N,typename T>
sparse_bispace<N> sparse_btensor<N,T>::get_bispace()
{
    return m_bispace;
}



template<size_t N,typename T>
std::string sparse_btensor<N,T>::str() const
{

    //Generate the loops for this tensor in slow->fast index order
    block_loop<0,1> outer(sequence<0,size_t>(),sequence<1,size_t>(0),sequence<0,bool>(),sequence<1,bool>(false));
    for(size_t i = 1; i < N; ++i)
    {
        outer.nest(sequence<0,size_t>(),sequence<1,size_t>(i),sequence<0,bool>(),sequence<1,bool>(false));
    }

    //TODO: Make all input sequences for bispaces use const, or some toher way of eliminating awkward const
    //from this method being const
    block_printer<double> bp;
    outer.run(bp,sequence<0,T*>(),
                 sequence<1,T*>(m_data),
                 sequence<0,sparse_bispace_generic_i*>(NULL),
                 sequence<1,sparse_bispace_generic_i*>((sparse_bispace<2>*)&m_bispace));
    return bp.str();
}

} // namespace libtensor

#endif /* SPARSE_BTENSOR_H */
