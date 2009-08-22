#ifndef BISPACE_DATA_H
#define BISPACE_DATA_H

#include <libtensor.h>

namespace libtensor {
	
/** \brief Base class for bispace data
 	
 	Stores the bispace<1> objects for 3 different types of indexes.
 	Derive from this class and implement a constructor which sets up 
 	the bispaces.
 	
	\ingroup libtensor_performance_tests
 **/
class bispace_data_i
{
protected:
	bispace<1> m_one, m_two, m_three;
public:
	/** \brief Constructor
	 **/
	bispace_data_i( size_t dim1, size_t dim2, size_t dim3 )
		: m_one(dim1), m_two(dim2), m_three(dim3) 
	{}
	
	/** \brief virtual destructor
	 **/ 
	virtual ~bispace_data_i() {}	
	
	/** \brief Return the first bispace<1> 
	 **/ 
	const bispace<1>& one() const { return m_one; }
					
	/** \brief Return the second bispace<1> 
	 **/ 
	const bispace<1>& two() const { return m_one; }
					
	/** \brief Return the third bispace<1> 
	 **/ 
	const bispace<1>& third() const { return m_one; }				
};	

/** \brief Bispace data for 2 large blocks per dimension 
 	\tparam O dimensions of the first index
 	\tparam V dimensions of the second index
 	
 	The third block index space will have dimension O+V  
 	
	\ingroup libtensor_performance_tests
 **/

template<size_t O, size_t V>
class large_blocks_data 
	: public bispace_data_i
{
public:
	/** \brief constructor
	 **/ 
	large_blocks_data();

	/** \brief virtual destructor
	 **/ 
	virtual ~large_blocks_data() {}	
	
};	

template<size_t O, size_t V>
large_blocks_data<O,V>::large_blocks_data()
	: bispace_data_i(O,V,O+V) 
{
	m_one.split(O/2);
	m_two.split(V/2);
	m_three.split((O+V)/2);
}
	
/** \brief Bispace data for many blocks per dimension 
 	\tparam O dimensions of the first index
 	\tparam V dimensions of the second index
 	\tparam BS average block size per dimension 
 	
 	The third block index space will have dimension O+V  
 	
	\ingroup libtensor_performance_tests
 **/

template<size_t O, size_t V, size_t BS>
class arbitrary_blocks_data 
	: public bispace_data_i
{
public:
	/** \brief constructor
	 **/ 
	arbitrary_blocks_data();

	/** \brief virtual destructor
	 **/ 
	virtual ~arbitrary_blocks_data() {}	
	
};	

template<size_t O, size_t V, size_t BS>
arbitrary_blocks_data<O,V,BS>::arbitrary_blocks_data()
	: bispace_data_i(O,V,O+V) 
{
	size_t pos=BS, nblocks=O/BS;
	for ( size_t i=0; i<nblocks; i++ ) { m_one.split(pos); pos+=BS;	}
	
	pos=BS; nblocks=V/BS;
	for ( size_t i=0; i<nblocks; i++ ) { m_two.split(pos); pos+=BS; }
	
	pos=BS;	nblocks=(O+V)/BS;
	for ( size_t i=0; i<nblocks; i++ ) { m_three.split(pos); pos+=BS; }
}


} // namespace libtensor


#endif // BISPACE_DATA_H
