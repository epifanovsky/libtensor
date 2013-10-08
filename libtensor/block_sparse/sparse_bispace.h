#ifndef LIBTENSOR_SPARSE_BISPACE_H
#define LIBTENSOR_SPARSE_BISPACE_H

#include <vector>
#include "../defs.h"
#include "../core/sequence.h"

namespace libtensor {

/* Forward declarations to enable a generic interface to all sparse_bispace degrees
 *
 */
template<size_t N>
class sparse_bispace;

//Non-templated sparse_bispace interface
//Necessary so that functions can be passed variable numbers of sparse_bispaces of different dimensions 
class sparse_bispace_generic_i {
public:
    virtual sparse_bispace<1> operator[](size_t idx) const = 0; 
    virtual size_t get_order() const = 0;
};

template<size_t N>
class sparse_bispace : public sparse_bispace_generic_i {
public: 
    static const char *k_clazz; //!< Class name
private:

    //Special case for creating a 2d space from 2 1d spaces
    sparse_bispace(const sparse_bispace<1>& spb_1,const sparse_bispace<1>& spb_2);
    std::vector< sparse_bispace<1> > m_subspaces;  
public:

    /** \brief Constructs a multi-dimensional space from a set of 1d subspaces 
     * INTERNAL USE ONLY - USE THE OPERATORS '|' and '>' is preferred 
     **/
    sparse_bispace(const std::vector< sparse_bispace<1> > &one_subspaces,const std::vector< sparse_bispace<1> > &two_subspaces);

    //TODO Delete this!!!
    /** \brief Returns the number of non-zero elements in this sparse bispace
     **/
    size_t get_nnz() const;

    /** \brief Combines the two operands to produce a new space
     **/
    template<size_t M> 
    sparse_bispace<N+M> operator|(const sparse_bispace<M>& rhs);

    /** \brief Retrieves the appropriate index subspace of this multidimensional space
        \throw out_of_bounds If an inappropriate index is specified 
     **/
    sparse_bispace<1> operator[](size_t  idx) const
        throw(out_of_bounds);

    /** \brief Returns the order of this bispace 
     **/
    size_t get_order() const;

    //TODO '>' operator for sparsity
    
    //Necessary to create 2d space from 2 1d spaces
    friend class sparse_bispace<1>;

    /** \brief Returns whether this object is equal to another of the same dimension. 
     *         Two N-D spaces are equal if all of their subspaces are equal and in the same order  
     **/
    bool operator==(sparse_bispace<N>& rhs) const;

    /** \brief Returns whether this object is not equal to another of the same dimension. 
     *         Two N-D spaces are equal if all of their subspaces are equal and in the same order
     **/
    bool operator!=(sparse_bispace<N>& rhs) const;
};

template<size_t N> 
sparse_bispace<N>::sparse_bispace(const std::vector< sparse_bispace<1> > &one_subspaces,const std::vector< sparse_bispace<1> > &two_subspaces)
{
    m_subspaces.reserve(one_subspaces.size() + two_subspaces.size());
    for(int i = 0; i < one_subspaces.size(); ++i)
    {
        m_subspaces.push_back(one_subspaces[i]);
    }
    for(int i = 0; i < two_subspaces.size(); ++i)
    {
        m_subspaces.push_back(two_subspaces[i]);
    }
}

template<size_t N>
sparse_bispace<N>::sparse_bispace(const sparse_bispace<1>& spb_1,const sparse_bispace<1>& spb_2)
{
    m_subspaces.push_back(spb_1);
    m_subspaces.push_back(spb_2);
}

template<size_t N>
size_t sparse_bispace<N>::get_nnz() const
{
    //TODO: Case where all elements are zero, fully sparse tensors????
    size_t nnz = 1;
    for(int i = 0; i < N; ++i)
    {
        nnz *= m_subspaces[i].get_dim(); 
    }
    return nnz;
}

template<size_t N> template<size_t M> 
sparse_bispace<N+M> sparse_bispace<N>::operator|(const sparse_bispace<M>& rhs)
{
    return sparse_bispace<N+M>(this->m_subspaces,rhs.m_subspaces);
}

//TODO: Should make these check (N-1) instead of m_subspaces.size()
template<size_t N>
sparse_bispace<1> sparse_bispace<N>::operator[](size_t idx) const
{
    if(idx > (m_subspaces.size() - 1))
    {
        throw out_of_bounds(g_ns,"sparse_bispace<N>","operator[](...)",
                __FILE__,__LINE__,"idx > (# of subspaces - 1) was specified"); 
    }
    return m_subspaces[idx];
}

template<size_t N>
size_t sparse_bispace<N>::get_order() const
{
    return N;
}

template<size_t N>
bool sparse_bispace<N>::operator==(sparse_bispace<N>& rhs) const
{
    for(int i = 0; i < N; ++i)
    {
        if(m_subspaces[i] != rhs.m_subspaces[i])
        {
            return false;
        }
    }
    return true;
}

template<size_t N>
bool sparse_bispace<N>::operator!=(sparse_bispace<N>& rhs) const
{
    return !(*this == rhs);
}

template<size_t N>
const char *sparse_bispace<N>::k_clazz = "sparse_bispace<N>";

/**  One-dimensional sparse block index space
 **/
template<>
class sparse_bispace<1> : public sparse_bispace_generic_i {
private:
    size_t m_dim; //!< Number of elements
    std::vector<size_t> m_abs_indices; //!< Block absolute starting indices
public: 

    /** \brief Creates the sparse block %index space with a given dimension
        \param dim Number of elements in this space.
     **/
    sparse_bispace(size_t dim);
    
    /** \brief Returns the dimension of the block index space 
     **/
    size_t get_dim() const;

    /** \brief Returns the number blocks into which this space has been split 
     **/
    size_t get_n_blocks() const;

    /** \brief Splits this space into blocks with offsets starting at offsets
               in split_points. First block always starts at zero
        \param split_points Iterable container of absolute indices where each block should start 
        \throw out_of_bounds If a split_point value exceeds the index limits, or if a zero length vector is passed 
     **/
    void split(const std::vector<size_t> &split_points)
        throw(out_of_bounds);

    /** \brief Returns the size of the block with block index block_idx
        \throw out_of_bounds If (# of blocks  - 1) < block_idx || block_idx < 0
     **/
    size_t get_block_size(size_t block_idx) const 
        throw(out_of_bounds);

    /** \brief Returns the absolute starting index of the block with block index block_idx
        \throw out_of_bounds If (# of blocks  - 1) < block_idx < 0
     **/
    size_t get_block_abs_index(size_t block_idx) const 
        throw(out_of_bounds);

    /** \brief Returns a 2d sparse_bispace composed of the two arguments
     **/
    sparse_bispace<2> operator|(const sparse_bispace<1>& rhs);

    /** \brief Returns a copy of this object 
        \throw out_of_bounds If an inappropriate index is specified 
     **/
    sparse_bispace<1> operator[](size_t  idx) const
        throw(out_of_bounds);

    /** \brief Returns the order of this bispace 
     **/
    size_t get_order() const;

    /** \brief Returns whether this object is equal to another. 
     *         Equality is defined to be the same dimension and block splitting pattern
     **/
    bool operator==(sparse_bispace<1>& rhs) const;

    /** \brief Returns whether this object is not equal to another. 
     *         Equality is defined to be the same dimension and block splitting pattern
     **/
    bool operator!=(sparse_bispace<1>& rhs) const;
};


inline sparse_bispace<1>::sparse_bispace(size_t dim) : m_dim(dim)
{ 
    m_abs_indices.push_back(0);
} 

inline size_t sparse_bispace<1>::get_dim() const
{
    return m_dim;
}

inline size_t sparse_bispace<1>::get_n_blocks() const
{
    return m_abs_indices.size();
}

inline void sparse_bispace<1>::split(const std::vector<size_t> &split_points)
{
    if(split_points.size() < 1 || split_points.size() > (m_dim - 1))
    {
        throw out_of_bounds(g_ns,"sparse_bispace<1>","split(...)",
                __FILE__,__LINE__,"Must have 1 <= # of split points <= dim - 1"); 
    }

    for(int i = 0; i < split_points.size(); ++i)
    {
        size_t split_point = split_points[i];
        if(split_point > (m_dim - 1))
        {
            throw out_of_bounds(g_ns,"sparse_bispace<1>","split(...)",
                    __FILE__,__LINE__,"Split point indices cannot exceed (dim - 1)"); 
        }
        else if(split_point <= m_abs_indices.back())
        {
            throw out_of_bounds(g_ns,"sparse_bispace<1>","split(...)",
                    __FILE__,__LINE__,"Split point indices must be strictly increasing"); 
        }
        m_abs_indices.push_back(split_point);
    }
}

inline size_t sparse_bispace<1>::get_block_size(size_t block_idx) const 
{
    if(block_idx > (m_abs_indices.size() - 1))
    {
        throw out_of_bounds(g_ns,"sparse_bispace<1>","get_block_size(size_t block_idx)",
                __FILE__,__LINE__,"Cannot pass block_idx greater than (# of blocks - 1)"); 
    }
    else if(block_idx == (m_abs_indices.size() - 1))
    {
        return m_dim - m_abs_indices.back(); 
    }
    else
    {
        return m_abs_indices[block_idx + 1] - m_abs_indices[block_idx];
    }
}


inline size_t sparse_bispace<1>::get_block_abs_index(size_t block_idx) const 
{
    if(block_idx > (m_abs_indices.size() - 1))
    {
        throw out_of_bounds(g_ns,"sparse_bispace<1>","get_block_abs_index(size_t block_idx)",
                __FILE__,__LINE__,"Cannot pass block_idx greater than (# of blocks - 1)"); 
    }
    return m_abs_indices[block_idx];
}

inline sparse_bispace<2> sparse_bispace<1>::operator|(const sparse_bispace<1>& rhs)
{
    return sparse_bispace<2>(*this,rhs);
}
    /** \brief Returns a copy of this object 
        \throw out_of_bounds If an inappropriate index is specified 
     **/
inline sparse_bispace<1> sparse_bispace<1>::operator[](size_t idx) const
{
    if(idx != 0)
    {
        throw out_of_bounds(g_ns,"sparse_bispace<1>","operator[](...)",
                __FILE__,__LINE__,"Invalid subspace index specified (can only specify 0"); 
    }
    return *this;
}

inline size_t sparse_bispace<1>::get_order() const
{
    return 1;
}


inline bool sparse_bispace<1>::operator==(sparse_bispace<1>& rhs) const
{
    return (this->m_dim == rhs.m_dim) && (this->m_abs_indices == rhs.m_abs_indices);
}

inline bool sparse_bispace<1>::operator!=(sparse_bispace<1>& rhs) const
{
    return ! (*this == rhs);
}

} // namespace libtensor

#endif // LIBTENSOR_SPARSE_BISPACE_H
