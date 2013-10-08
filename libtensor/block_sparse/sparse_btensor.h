#ifndef SPARSE_BTENSOR_H
#define SPARSE_BTENSOR_H

#include "sparse_bispace.h"
#include <sstream>
#include <string>

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

    //If a memory location with values to load was specified, load them in
    //if(mem != NULL)
    //{
        //if(! already_block_major)
        //{
        //}
    //}
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


#if 0
//NEED BLOCK MAJOR!!!
//TODO: Replace with template that takes loop list and function pointers and returns the result of the operation 
//Something like template<typename T, Functor<T> >, where a Functor<T> returns a type T...then can use for equality etc 
//The functor interface must always take three tensor block pointers and block size vector for each tensor...then can 
//use for contractions, transpose, equality, w/e. Just set the appropriate one to null for generalizing
/* Used to recursively determine tensor equality
 */
template<size_t N,typename T>
bool _equality_recurse(const sparse_bispace<N>& the_bispace,T* lhs,T* rhs, size_t cur_subspace_idx,size_t offset) const
{
    //Base case
    if(cur_subspace_idx == (N -1))
    {
        for(size_t i = 0; i < the_bispace[cur_subspace_idx].size(); ++i)
        {
            if( (lhs+i) != (rhs+i))
            {
                return false;
            }
        }
        return true;
    }
    else
    {

        //TODO: Support sparsity - can't just use get_dim
        size_t inner_size = 1;
        for(size_t i = cur_subspace_idx+1; i < N; ++i)
        {
            inner_size *= the_bispace[i].get_dim();
        }

        for(size_t i = 0; i < the_bispace[cur_subspace_idx].get_dim(); ++i)
        {
            bool sub_equality = _equality_recurse(bispaces,cur_bispace_idx+1,offset);
            if(!sub_equality)
            {
                return false;
            }
            offset += inner_size;
        }
        return true;
    }
}

template<size_t N,typename T>
bool sparse_btensor<N,T>::operator==(const sparse_btensor<N,T>& rhs) const
{
    if(m_bispace != rhs.m_bispace)
    {
        return false;
    }
    _equality_recurse(m_bispace,m_data,rhs.m_data,0,0);
}
#endif

//TODO: re-do using sequences for compile-time size checking
template<typename T>
void _str_intra_block(std::stringstream& ss,const T* block,std::vector<size_t>& block_dims,size_t dim_idx)
{
    //Base case
    if(dim_idx == (block_dims.size() - 1))
    {
        for(int i = 0; i < block_dims.back(); ++i)
        {
            ss << " " << block+i; 
        }
        ss << "\n";
    }
    else
    {
        size_t inner_size = 1;
        for(size_t i = dim_idx +1; i < block_dims.size(); ++i)
        {
            inner_size *= block_dims[i];
        }
        for(size_t i = 0; i < block_dims[dim_idx]; ++i)
        {
            _str_intra_block(ss,block,block_dims,dim_idx+1);
            block += block_dims[dim_idx]*inner_size;
        }
    }
}

//TODO: SOME LOOP-LIST BASED ROUTINE FOR THIS!!!!
//TODO: General inter-block routine that calls a block_functor...
//I CAN GENERALIZE NO DOUBT as long as the # of tensors is called using a standard order
//But DONT GENERALIZE UNTIL I HAVE 3 - RULE OF THREE
//print,equality,contraction, then generalize and make them use the same sparse block tensor operation class
//Ask evgeny for advice
//TODO: Sparse support
template<size_t N,typename T>
void _str_inter_block(std::stringstream& ss,const sparse_bispace<N>& the_bispace,size_t subspace_idx,const T* block_major_data,size_t offset,std::vector<size_t>& block_dims)
{
    //Base case: now do intra-block
    if(subspace_idx == (N-1))
    {
        _str_intra_block(ss,block_major_data,block_dims,0);
        ss << "----\n";
    }
    else
    {
        size_t inner_size = 1;
        for(size_t i = subspace_idx+1; i < N; ++i)
        {
            inner_size *= the_bispace[i].get_dim();
        }

        sparse_bispace<1> cur_subspace = the_bispace[subspace_idx];
        for(size_t block_idx = 0; block_idx < cur_subspace.get_n_blocks(); ++block_idx)
        {
            size_t cur_block_size = cur_subspace.get_block_size(block_idx);
            block_dims[subspace_idx] = cur_block_size;
            _str_inter_block(ss,the_bispace,subspace_idx+1,block_major_data+offset,offset,block_dims);
            offset += cur_subspace.get_block_size(block_idx)*inner_size; 
            ss << "####\n";
        }
    }
}

template<size_t N,typename T>
std::string sparse_btensor<N,T>::str() const
{
    std::stringstream ss; 
    std::vector<size_t>  block_dims(N);
    _str_inter_block(ss,m_bispace,0,m_data,0,block_dims);
    return ss.str();
}

} // namespace libtensor

#endif /* SPARSE_BTENSOR_H */
