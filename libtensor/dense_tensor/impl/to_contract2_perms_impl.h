#ifndef LIBTENSOR_TO_CONTRACT2_PERMS_IMPL_H
#define LIBTENSOR_TO_CONTRACT2_PERMS_IMPL_H

#include "../to_contract2_perms.h"
#include "../../core/sequence.h"
#include <algorithm> //to use std::swap

namespace libtensor {


template<size_t N, size_t M, size_t K>
void to_contract2_perms<N, M, K>::make_perms(
    const contraction2<N, M, K> &contr, const dimensions<N + K> &dimsa,
    const dimensions<M + K> &dimsb, const dimensions<N + M> &dimsc) {

	using std::max;

	sequence<k_maxconn, size_t> new_conn; //!< Index connections
	new_conn = contr.get_conn();

//	std::cout << "initial connection:\n";
//	for(size_t i = 0; i < k_maxconn; i++) {
//		std::cout << new_conn[i] << " ";
//	}
//	std::cout << "\n";

	size_t permute_zone_a, permute_zone_b, permute_zone_c;
	permute_zone_a = permute_zone_b = permute_zone_c = 0;
	//first we have to separate contracted and non-contracted indexes
	//let's start from A
	//Is the last index in A a non-contracted index (index from C)?
	if (new_conn[k_orderc + k_ordera-1] < k_orderc) { //if yes
		//now all non-contracted indexes in A should be moved to the right and contracted - to the left.

		//let's find which index has to be moved to the right and define permuted zone
		for(size_t i = k_orderc+k_ordera-1; i > k_orderc ; i--) {
			//find first contracted index
			if( new_conn[i] >= k_orderc ) {
				//now find if there are any non-contracted indexes left
				for(size_t j = i - 1; j >= k_orderc; j-- ) {
					if( new_conn[j] < k_orderc ) {
						//move non-contracted index to the right by permuting it with the right most contracted index
						m_perma.permute(i-k_orderc, j-k_orderc);
						//HERE WE HAVE TO CHANGE THE CONNECTION IN CONN
						permute_conn(new_conn, i, j);
						//define permuted zone if it was not define
						//move permuted zone more to the left if required
						if( permute_zone_a < i-k_orderc) {
							permute_zone_a = i-k_orderc;
						}
						//now the right most contracted index is i-1
						i--;
					}
				} //end for
				//no need to go further
				break;
			} //end if
		}//end for //indexes in A moved to the right

	} else { //the last index in A is a contracted index (from B)
		//let's find which index has to be moved to the right and define permuted zone
		for(size_t i = k_orderc+k_ordera-1; i > k_orderc ; i--) {
			//find first contracted index (from C)
			if( new_conn[i] < k_orderc ) {
				//now find if there are any contracted (from B) indexes left
				for(size_t j = i - 1; j >= k_orderc; j-- ) {
					if( new_conn[j] >= k_orderc ) {
						//move non-contracted index to the right by permuting it with the right most contracted index
						m_perma.permute(i-k_orderc, j-k_orderc);
						//HERE WE HAVE TO CHANGE THE CONNECTION IN CONN
						permute_conn(new_conn, i, j);
						//define permuted zone if it was not define
						//move permuted zone more to the left if required
						if( permute_zone_a < i-k_orderc) {
							permute_zone_a = i-k_orderc;
						}
						//now the right most contracted index is i-1
						i--;
					}
				} //end for
				//no need to go further
				break;
			} //end if
		}//end for //indexes in A moved to the right

	} //end grouping indexes in A

//	std::cout << "after grouping A:\n";
//		for(size_t i = 0; i < k_maxconn; i++) {
//			std::cout << new_conn[i] << " ";
//		}
//		std::cout << "\n";
	//now we have to group all the indexes in C
	//is the last index in C from A or from B?
	if (new_conn[k_orderc -1] < k_orderc + k_ordera) { //last index is from A
		//last indexes in A and C are already compared - no need to check
		//let's find which index has to be moved to the right and define permuted zone
		for(size_t i = k_orderc-1; i > 0 ; i--) {
			//find first index from B
			if( new_conn[i] >= k_orderc + k_ordera ) {
				//now find if there are any indexes from A left
				for(size_t j = i - 1; j >= 0; j-- ) {
					if( new_conn[j] < k_orderc + k_ordera ) {
						//move non-contracted index to the right by permuting it with the right most contracted index
						m_perma.permute(i, j);
						//!!!!HERE WE HAVE TO CHANGE THE CONNECTION IN CONN
						permute_conn(new_conn, i, j);
						//define permuted zone if it was not define
						//move permuted zone more to the left if required
						if( permute_zone_c < i) {
							permute_zone_c = i;
						}
						//now the right most contracted index is i-1
						i--;
					}
				} //end for
				//no need to go further
				break;
			} //end if
		}//end for //indexes in C moved to the right

	} else {// if last index is from B
		//let's find which index has to be moved to the right and define permuted zone
//		std::cout << "TEST1\n";
		for(size_t i = k_orderc-1; i > 0 ; i--) {
			//find first index from A
			if( new_conn[i] < k_orderc + k_ordera ) {
				//now find if there are any indexes from B left
				for(int j = i - 1; j >= 0; j-- ) {
					if( new_conn[j] >= k_orderc + k_ordera ) {
//						std::cout << "TEST2\n";
//						std::cout << "i = " << i << ", j = " << j << ", new_conn[j] = " << new_conn[j] << "\n";
						//move non-contracted index to the right by permuting it with the right most contracted index
						m_perma.permute(i, j);
						//CHANGE THE CONNECTION IN CONN
						permute_conn(new_conn, i, j);
						//define permuted zone if it was not define
						//move permuted zone more to the left if required
						if( permute_zone_c < i) {
							permute_zone_c = i;
						}
						//now the right most contracted index is i-1
						i--;
					}
				} //end for
				//no need to go further
				break;
			} //end if
		}//end for //indexes in C moved to the right
	} //now all the indexes in C grouped

//	std::cout << "after grouping C:\n";
//			for(size_t i = 0; i < k_maxconn; i++) {
//				std::cout << new_conn[i] << " ";
//			}
//			std::cout << "\n";

	//now we have to group indexes in B
	//Is the last index in B from C (non-contracted)?
	if (new_conn[k_maxconn-1] < k_orderc) { //if yes
		//now all non-contracted indexes in B should be moved to the right and contracted - to the left.

		//let's find which index has to be moved to the right and define permuted zone
		for(size_t i = k_maxconn-1; i > k_orderc + k_ordera ; i--) {
			//find first contracted index
			if( new_conn[i] >= k_orderc ) {
				//now find if there are any non-contracted indexes left
				for(size_t j = i - 1; j >= k_orderc + k_ordera; j-- ) {
					if( new_conn[j] < k_orderc ) {
						//move non-contracted index to the right by permuting it with the right most contracted index
						m_perma.permute(i-k_orderc-k_ordera, j-k_orderc-k_ordera);
						//!!!!HERE WE HAVE TO CHANGE THE CONNECTION IN CONN
						permute_conn(new_conn, i, j);
						//define permuted zone if it was not define
						//move permuted zone more to the left if required
						if( permute_zone_b < i-k_orderc-k_ordera) {
							permute_zone_b = i-k_orderc-k_ordera;
						}
						//now the right most contracted index is i-1
						i--;
					}
				} //end for
				//no need to go further
				break;
			} //end if
		}//end for //indexes in B moved to the right

	} else { //the last index in B a contracted index (from A)
		//let's find which index has to be moved to the right and define permuted zone
		for(size_t i = k_maxconn-1; i > k_orderc + k_ordera ; i--) {
			//find first contracted index (from C)
			if( new_conn[i] < k_orderc ) {
				//now find if there are any contracted (from A) indexes left
				for(size_t j = i - 1; j >= k_orderc + k_ordera; j-- ) {
					if( new_conn[j] >= k_orderc ) {
						//move non-contracted index to the right by permuting it with the right most contracted index
						m_perma.permute(i-k_orderc-k_ordera, j-k_orderc-k_ordera);
						//!!!!HERE WE HAVE TO CHANGE THE CONNECTION IN CONN
						permute_conn(new_conn, i, j);
						//define permuted zone if it was not define
						//move permuted zone more to the left if required
						if( permute_zone_b < i-k_orderc-k_ordera) {
							permute_zone_b = i-k_orderc-k_ordera;
						}
						//now the right most contracted index is i-1
						i--;
					}
				} //end for
				//no need to go further
				break;
			} //end if
		}//end for //indexes in B moved to the right
	} //end grouping indexes in B

//	std::cout << "connection after grouping:\n";
//		for(size_t i = 0; i < k_maxconn; i++) {
//			std::cout << new_conn[i] << " ";
//		}
//		std::cout << "\n";

	//Finally we grouped all indexes in all tensors!
	//Now we can reorder them such that they match each other
	//Let's start comparing A and C
	//check the last indexes first

	//difference between corresponding indexes in A and C, depends on the last index from C
	//if last index in A is from C shift_a = 0, otherwise K (number of indexes from B in A)
	size_t shift_a = (new_conn[k_orderc + k_ordera - 1] < k_orderc) ? 0 : K;
	//difference between corresponding indexes in A and C, depends on the last index from C
	size_t shift_c = (new_conn[k_orderc-1] < k_orderc+k_ordera) ? k_ordera - shift_a: k_ordera + M - shift_a;
	for (size_t i = k_orderc+k_ordera - shift_a - 1; i > k_orderc+ K - shift_a; i--) {
		if(new_conn[i] != i - shift_c) { //index in C doesn't correspond to index in A
			//which is easier to permute: A or C?
			if( does_permute_first(dimsa, dimsc, i - k_orderc, i - shift_c, permute_zone_a, permute_zone_b)) { //A
				m_perma.permute(i - k_orderc, new_conn[i - shift_c] - k_orderc);
				//HERE WE HAVE TO CHANGE THE CONNECTION IN CONN
				permute_conn(new_conn, i, new_conn[i - shift_c]);
				//define permuted zone if it was not define
				permute_zone_a = max(i-k_orderc, permute_zone_a);
			} else { //permute C
				m_permc.permute(i - shift_c, new_conn[i]);
				permute_conn(new_conn, i - shift_c, new_conn[i]);
				permute_zone_c = max(i - shift_c, permute_zone_c);
			}
		} //else if index in A correspond to the index in C we can go to the next index
	} //end for: now the order of indexes in A and C is the same

//	std::cout << "after ac connection:\n";
//	for(size_t i = 0; i < k_maxconn; i++) {
//			std::cout << new_conn[i] << " ";
//		}
//		std::cout << "\n";
	//Now we can compare indexes in A and B

	//difference between corresponding indexes in A and B, depends on the last index from C in A
	//if last index in A is from C shift_ab = N (number of indexes from C in A), otherwise 0
	size_t shift_ab = (new_conn[k_orderc + k_ordera - 1] < k_orderc) ? N : 0;
	//difference between corresponding indexes in A and B, depends on the last index from B
	//if last index in B is from A then shift_b = k_orderb + shift_ab
	size_t shift_b = (new_conn[k_maxconn-1] >= k_orderc) ? k_orderb + shift_ab: k_orderb + shift_ab - M;
	for (size_t i = k_orderc + k_ordera - shift_ab - 1; i > k_orderc + N - shift_ab; i--) {
		if(new_conn[i] != i + shift_b) { //index in B doesn't correspond to index in A
			//which is easier to permute: A or B?
			if( does_permute_first(dimsa, dimsb, i - k_orderc, i + shift_b - k_ordera -k_orderc, permute_zone_a, permute_zone_b )) { //A
				m_perma.permute(i - k_orderc, new_conn[i + shift_b] - k_orderc);
				permute_conn(new_conn, i, new_conn[i + shift_b]);
				//define permuted zone if it was not define
				permute_zone_a = max(i-k_orderc, permute_zone_a);
			} else { //permute B
				m_permb.permute(i + shift_b - k_ordera - k_orderc, new_conn[i] - k_ordera -k_orderc);
				permute_conn(new_conn, i + shift_b, new_conn[i]);
				permute_zone_b = max(i + shift_b - k_ordera -k_orderc, permute_zone_b);
			}
		} //else if index in A correspond to the index in B we can go to the next index
	} //end for: indexes in A and B are in the same order.
//	std::cout << "after ab connection:\n";
//	for(size_t i = 0; i < k_maxconn; i++) {
//			std::cout << new_conn[i] << " ";
//		}
//		std::cout << "\n";


	//Now we have only to compare indexes in C from B

	//difference between corresponding indexes in C and B, depends on the last index from B in C
	//if last index in C is from B shift_cb = 0, otherwise N (number of indexes from A in C)
	size_t shift_cb = (new_conn[k_orderc - 1] >= k_orderc + k_ordera) ? 0 : N;
	//difference between corresponding indexes in C and B, depends on the last index from B
	//if last index in B is from C then shift_b = - k_order_b
	shift_b = (new_conn[k_maxconn-1] < k_orderc) ? k_ordera + k_orderb + shift_cb: k_ordera + k_orderb + shift_a - K;
	for (size_t i = k_orderc - shift_cb - 1; i > N - shift_cb; i--) {
		if(new_conn[i] != i + shift_b) { //index in B doesn't correspond to index in C
			//which is easier to permute: C or B?
			if( does_permute_first(dimsc, dimsb, i , i + shift_b - k_ordera -k_orderc, permute_zone_a, permute_zone_b )) { //A
				m_permc.permute(i, new_conn[i + shift_b]);
				permute_conn(new_conn, i, new_conn[i + shift_b]);
				//define permuted zone if it was not define
				permute_zone_c = max(i, permute_zone_a);
			} else { //permute B
				m_permb.permute(i + shift_b - k_ordera - k_orderc, new_conn[i] - k_ordera -k_orderc);
				permute_conn(new_conn, i + shift_b, new_conn[i]);
				permute_zone_b = max(i + shift_b - k_ordera -k_orderc, permute_zone_b);
			}
		} //else if index in A correspond to the index in B we can go to the next index
	} //end fo

	//!!!WE ARE FINALLY DONE!
//	std::cout << "final connection:\n";
//	for(size_t i = 0; i < k_maxconn; i++) {
//		std::cout << new_conn[i] << " ";
//	}
//	std::cout << "\n";

} //end make_perms



template<size_t N, size_t M, size_t K>
	template<size_t A, size_t B>
bool to_contract2_perms<N, M, K>::does_permute_first(const dimensions<A> &dimsa,
	    const dimensions<B> &dimsb, size_t perm_indexa, size_t perm_indexb, size_t permute_zone_a, size_t permute_zone_b) {
	//how many index elements should be continues in memory to achieve speedup
//	const size_t linear = 4;
//	//how many times faster is copying linear
//	const size_t speedup = 4;

	if (perm_indexa < 1 || perm_indexb < 1) {
		std::cout << "Incorrect perm indexes in does_permute_first";
		//throw an exception
	}

//	std::cout << "Permute zones: a = " << permute_zone_a << ", b = " << permute_zone_b << "\n";
//	std::cout << "Permute index: a = " << perm_indexa << ", b = " << perm_indexb << "\n";

	//check if indexes that has to be permuted are in the permuted zone
//	if (perm_indexa <= permute_zone_a || (permute_zone_a && dimsa.get_increment(perm_indexa) > 4 )) {
//		return true;
//	} else if (perm_indexb <= permute_zone_b || (permute_zone_b && dimsb.get_increment(perm_indexb) > 4) ) {
//		return false;
//	}

	if (perm_indexa <= permute_zone_a ) {
		return true;
	} else if (perm_indexb <= permute_zone_b ) {
		return false;
	}

	//size of the indexes that has to be permuted
	size_t sizea, sizeb;

	sizea = get_permute_cost(dimsa, perm_indexa) - get_permute_cost(dimsa, permute_zone_a);
	sizeb = get_permute_cost(dimsb, perm_indexb) - get_permute_cost(dimsb, permute_zone_b);
	//size of the indexes that will not be permuted
//	size_t nonperm_sizea, nonperm_sizeb;
//
//	sizea = dimsa.get_size();// SHOULD IT BE DEVIDED BY THE NUMBER OF INDEXES IN PERMUTED ZONE SINCE THAY HAVE TO BE PERMUTED ANYWAY?
//	nonperm_sizea = dimsa.get_increment(perm_indexa - 1);
//	//copying of the 'linear' number of sequential indexes takes 'speedup' times less than copying one index
//	if ( nonperm_sizea >= linear )
//		sizea = (sizea + speedup - 1) / speedup;
//	sizeb = dimsb.get_size();
//	nonperm_sizeb = dimsb.get_increment(perm_indexb - 1);
//	//copying of the 'linear' number of sequential indexes takes 'speedup' times less than copying one index
//	if ( nonperm_sizeb >= linear )
//		sizeb = (sizeb + speedup - 1) / speedup;

//	std::cout << "Size a = " << sizea << ", b = " << sizeb << "\n";
	return (sizea <= sizeb);
}

template<size_t N, size_t M, size_t K>
	template<size_t A>
size_t to_contract2_perms<N, M, K>::get_permute_cost(const dimensions<A> &dimsa,
	    size_t perm_indexa) {
	//how many index elements should be continues in memory to achieve speedup
	const size_t linear = 4;
	//how many times faster is copying linear
	const size_t speedup = 4;

	if (perm_indexa < 1) //nothing to permute
		return 0;
	//size of the indexes that has to be permuted
	size_t sizea;
	//size of the indexes that will not be permuted
	size_t nonperm_sizea;

	sizea = dimsa.get_size();
	nonperm_sizea = dimsa.get_increment(perm_indexa);
	//copying of the 'linear' number of sequential indexes takes 'speedup' times less than copying one index
	if ( nonperm_sizea >= linear )
		sizea = (sizea + speedup - 1) / speedup;
	return sizea;
}

template<size_t N, size_t M, size_t K>
void to_contract2_perms<N, M, K>::permute_conn(sequence<k_maxconn, size_t> &conn, size_t i, size_t j) {
	sequence<k_maxconn, size_t> old_conn(conn);
	conn[i] = old_conn[j];
	conn[j] = old_conn[i];
	conn[old_conn[i]] = j;
	conn[old_conn[j]] = i;
}

} // namespace libtensor

#endif // LIBTENSOR_TO_CONTRACT2_PERMS_IMPL_H

