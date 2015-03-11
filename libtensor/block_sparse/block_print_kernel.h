/*
 * block_print_kernel.h
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#ifndef BLOCK_PRINT_KERNEL_H_
#define BLOCK_PRINT_KERNEL_H_

#include "block_kernel_i.h"
#include <sstream>

namespace libtensor
{

template<typename T>
class block_print_kernel: public libtensor::block_kernel_i<block_print_kernel<T>,T>
{
private:
    std::stringstream m_ss;
    static const char *k_clazz; //!< Class name

    //Used to recursively traverse block to print individual elements
    void _process_dimension(const T* data_ptr,const dim_list& dims,size_t offset = 0,size_t dim_idx = 0);
public:

	void operator()(const std::vector<T*>& ptrs, const std::vector< dim_list >& dim_lists);
    std::string str() const { return m_ss.str(); };
};


} /* namespace libtensor */

template<typename T>
const char* libtensor::block_print_kernel<T>::k_clazz = "block_print_kernel<T>";

template<typename T>
void libtensor::block_print_kernel<T>::_process_dimension(const T* data_ptr,const dim_list& dims,size_t offset,size_t dim_idx)
{
    //Base case
    if(dim_idx == (dims.size() - 1))
    {
        const T* inter_data_ptr = data_ptr + offset;
        for(int i = 0; i < dims.back(); ++i)
        {
            m_ss << ' ' << *inter_data_ptr;
            inter_data_ptr += 1;
        }
        m_ss << std::endl;
        return;
    }
    else
    {
        //Delimit blocks of varying dimensions by the corresponding number of newlines
        //But must skip first block for aesthetics
        if(offset != 0)
        {
            m_ss << std::endl;
        }
        size_t inner_size = 1;
        for(int i = dim_idx+1; i < dims.size(); ++i)
        {
            inner_size *= dims[i];
        }
        for(int i = 0; i < dims[dim_idx]; ++i)
        {
            _process_dimension(data_ptr,dims,offset,dim_idx+1);
            offset += inner_size;
        }
    }
}

template<typename T>
void libtensor::block_print_kernel<T>::operator()(
		const std::vector<T*>& ptrs, const std::vector<dim_list>& dim_lists)
{
	if(ptrs.size() != 1 || ptrs.size() != dim_lists.size())
	{
		throw bad_parameter(g_ns, k_clazz,"block_print_kernel(...)",
				__FILE__, __LINE__, "incorrect number of pointers and dimension lists");
	}

    m_ss << "---\n";
    _process_dimension(ptrs[0],dim_lists[0]);
}

#endif /* BLOCK_PRINT_KERNEL_H_ */
