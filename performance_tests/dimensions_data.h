#ifndef DIMENSIONS_DATA_H
#define DIMENSIONS_DATA_H

namespace libtensor {

/** \brief interface for dimension data
 	
 	Stores the information to form 3 dimensions objects with different 
 	dimensions. Any class which inherits from this should implement a 
 	constructor to setup the indexes indA, indB and indC. 
 **/	
template<size_t N, size_t M, size_t K> 
class dimensions_data_i {
protected:
	index<N+M> indA;  //!< last index in an index_range to form dimA
	index<N+K> indB;  //!< last index in an index_range to form dimB
	index<M+K> indC;  //!< last index in an index_range to form dimC
public:
	virtual ~dimensions_data_i() {}
	dimensions<N+M> dimA(); 
	dimensions<N+K> dimB();
	dimensions<M+K> dimC();
};
	
template<size_t N, size_t M, size_t K, size_t Size> 
class dimensions_data : public dimensions_data_i<N,M,K> {
public:
	dimensions_data();		
	virtual ~dimensions_data() {}
};

template<size_t N, size_t M, size_t K> 
dimensions<N+M> 
dimensions_data_i<N,M,K>::dimA() 
{
	index_range<N+M> ir(index<N+M>(),indA);
	return dimensions<N+M>(ir);
}

template<size_t N, size_t M, size_t K> 
dimensions<N+K> 
dimensions_data_i<N,M,K>::dimB() 
{
	index_range<N+K> ir(index<N+K>(),indB);
	return dimensions<N+K>(ir);
}

template<size_t N, size_t M, size_t K> 
dimensions<M+K> 
dimensions_data_i<N,M,K>::dimC() 
{
	index_range<M+K> ir(index<M+K>(),indC);
	return dimensions<M+K>(ir);
}

template<size_t N, size_t M, size_t K, size_t Size> 
dimensions_data<N,M,K,Size>::dimensions_data()
{ 
	for ( size_t i=0; i<N+M; i++ ) dimensions_data::indA[i]=Size-1;
	for ( size_t i=0; i<N+K; i++ ) dimensions_data::indB[i]=Size-1;
	for ( size_t i=0; i<M+K; i++ ) dimensions_data::indC[i]=Size-1;
}




} // namespace libtensor

#endif // DIMENSIONS_DATA_H 
