#ifndef RUNTIME_PERMUTATION_H
#define RUNTIME_PERMUTATION_H

#include <vector>
#include <algorithm>
#include "../core/sequence.h"

//TODO: Un-inline and take this out of header!!!!!
namespace libtensor {

/* This class exists because at compile time, I may not know the degree of the tree that I will need to permute
 *
 * As such, I cannot use templates
 */
class runtime_permutation
{
private:
    //Values are the DESTINATION index for the given index
    std::vector<size_t> m_idx;

public:
    //Constructs an identity permutation of a given size
    runtime_permutation(size_t N);

    //Construct from a permutation vector
    runtime_permutation(const std::vector<size_t>& vec);

    template<size_t N,typename T> 
    void apply(sequence<N,T>& seq) const;

    //Send i to j, j to i
    void permute(size_t i,size_t j);

    size_t get_order() const { return m_idx.size(); }

    bool operator==(const runtime_permutation& rhs) const;
    bool operator!=(const runtime_permutation& rhs) const;

    size_t operator[](size_t idx) const { return m_idx[idx]; }
} ;

inline runtime_permutation::runtime_permutation(size_t N) : m_idx(N)
{
    for(size_t i = 0; i < N; ++i) m_idx[i] = i;
}
//Construct from a permutation vector
inline runtime_permutation::runtime_permutation(const std::vector<size_t>& vec)
{
    //No duplicates?
    std::vector<size_t> processed;
    for(size_t i = 0; i < vec.size(); ++i)
    {
        if(!std::binary_search(processed.begin(),processed.end(),vec[i]))
        {
            processed.push_back(vec[i]);
            m_idx.push_back(vec[i]);
        }
        else
        {
            throw bad_parameter(g_ns,"runtime_permutation","runtime_permutation(...)",
                __FILE__,__LINE__,"duplicate entries not allowed"); 
        }
    }
}

template<size_t N,typename T> 
void runtime_permutation::apply(sequence<N,T>& seq) const 
{
    sequence<N, T> buf(seq);
    for(size_t i = 0; i < N; i++) seq[m_idx[i]] = buf[i];
}

inline void runtime_permutation::permute(size_t i,size_t j)
{
    std::swap(m_idx[i],m_idx[j]);
}

inline bool runtime_permutation::operator==(const runtime_permutation& rhs) const
{
    if(m_idx.size() != rhs.m_idx.size())
    {
        return false;
    }

    for(size_t i = 0; i < m_idx.size(); ++i)
    {
        if(m_idx[i] != rhs.m_idx[i])
        {
            return false;
        }
    }
    return true;
}

inline bool runtime_permutation::operator!=(const runtime_permutation& rhs) const
{
    return !(*this == rhs);
}

} // namespace libtensor

#endif /* RUNTIME_PERMUTATION_H */
