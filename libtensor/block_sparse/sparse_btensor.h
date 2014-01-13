#ifndef SPARSE_BTENSOR_H
#define SPARSE_BTENSOR_H

#include <sstream>
#include "sparse_bispace.h"
#include "../iface/letter_expr.h"
#include "sparse_loop_list.h"
#include "block_load_kernel.h"
#include "block_print_kernel.h"
#include "labeled_sparse_btensor.h"

namespace libtensor {


//Forward declaration for operator()(...) 
template<size_t N,typename T>
class labeled_sparse_btensor;

template<size_t N,typename T = double>
class sparse_btensor {
public:
    static const char *k_clazz; //!< Class name
private:
    T* m_data_ptr;
    sparse_bispace<N> m_bispace;
public:
    /** \brief Constructs a sparse block tensor object and populates it with the entries from mem if specified
     **/
    sparse_btensor(const sparse_bispace<N>& the_bispace,T* mem = NULL,bool already_block_major = false);
    virtual ~sparse_btensor();

    //Copy constructor
    sparse_btensor(const sparse_btensor<N>& rhs) : m_bispace(rhs.m_bispace) { m_data_ptr = new T[m_bispace.get_nnz()]; memcpy(m_data_ptr,rhs.m_data_ptr,m_bispace.get_nnz()*sizeof(T)); }
    /** \brief Return the sparse_bispace defining this tensor 
     **/
    const sparse_bispace<N>& get_bispace() const; 

    /** \brief Compares the tensor to another
     *         Two sparse_btensors are equal if they have the same number of elements and all of those elements match
     **/
    bool operator==(const sparse_btensor<N,T>& rhs) const;
    bool operator!=(const sparse_btensor<N,T>& rhs) const;

    const T* get_data_ptr() const { return m_data_ptr; }
    
    /** \brief Returns a labeled_sparse_btensor object for use in expressions
     *
     */
    labeled_sparse_btensor<N,T> operator()(const letter_expr<N>& le);

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
    m_data_ptr = new T[size];

    //Create loops

    if(mem != NULL)
    {
        if(already_block_major)
        {
            memcpy(m_data_ptr,mem,the_bispace.get_nnz()*sizeof(T));
        }
        else
        {

        	std::vector<sparse_bispace_any_order> bispaces(1,m_bispace);
            std::vector<block_loop> loops;
            for(size_t i = 0; i < N; ++i)
            {
            	block_loop loop(bispaces);
            	loop.set_subspace_looped(0,i);
            	loops.push_back(loop);
            }
        	sparse_loop_list sll(loops);

            block_load_kernel<T> blk(m_bispace,mem);
            std::vector<T*> ptrs(1,m_data_ptr);
            sll.run(blk,ptrs);
        }
    }
}

template<size_t N,typename T>
sparse_btensor<N,T>::~sparse_btensor()
{
    delete [] m_data_ptr;
}

template<size_t N,typename T>
const sparse_bispace<N>& sparse_btensor<N,T>::get_bispace() const
{
    return m_bispace;
}



template<size_t N,typename T>
bool sparse_btensor<N,T>::operator==(const sparse_btensor<N,T>& rhs) const
{
    if(this->m_bispace.get_nnz() != rhs.m_bispace.get_nnz())
    {
        throw bad_parameter(g_ns, k_clazz,"operator==(...)",
                __FILE__, __LINE__, "tensors have different numbers of nonzero elements");
    }

    for(size_t i = 0; i < m_bispace.get_nnz(); ++i)
    {
        if(m_data_ptr[i] != rhs.m_data_ptr[i])
        {
            return false;
        }
    }
    return true;
}

template<size_t N,typename T>
bool sparse_btensor<N,T>::operator!=(const sparse_btensor<N,T>& rhs) const
{
    return !(*this == rhs);
}


template<size_t N,typename T>
labeled_sparse_btensor<N,T> sparse_btensor<N,T>::operator()(const letter_expr<N>& le)
{
    return labeled_sparse_btensor<N,T>(*this,le);
}

template<size_t N,typename T>
std::string sparse_btensor<N,T>::str() const
{

    //Generate the loops for this tensor in slow->fast index order
	std::vector<sparse_bispace_any_order> bispaces(1,m_bispace);

    std::vector<block_loop> loops;
	for(size_t i = 0; i < N; ++i)
	{
		block_loop loop(bispaces);
		loop.set_subspace_looped(0,i);
		loops.push_back(loop);
	}
	sparse_loop_list sll(loops);

	block_print_kernel<T> bpk;
	std::vector<T*> ptrs(1,m_data_ptr);
	sll.run(bpk,ptrs);
    return bpk.str();
}

} // namespace libtensor

#endif /* SPARSE_BTENSOR_H */
