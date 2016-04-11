/*
 * Copyright (c) 2014-2016 Ilya Kaliman <ilya.kaliman@gmail.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef HAVE_BITSTRING_H
#include <bitstring.h>
#else
#include "compat/bitstring.h"
#endif

#ifdef HAVE_TREE_H
#include <sys/tree.h>
#else
#include "compat/tree.h"
#endif

#include "alloc.h"

#define XM_PAGE_SIZE (512ULL * 1024)
#define XM_GROW_SIZE (256ULL * 1024 * 1024 * 1024)

struct block {
	uintptr_t               data_ptr;
	size_t                  size_bytes;
	RB_ENTRY(block)         entry;
};

RB_HEAD(tree, block);

struct xm_allocator {
	int                     fd;
	char                   *path;
	size_t                  file_bytes;
	bitstr_t               *pages;
	pthread_mutex_t         mutex;
	struct tree             blocks;
};

static int
tree_cmp(const struct block *a, const struct block *b)
{
	return (a->data_ptr == b->data_ptr ? 0 :
	    a->data_ptr < b->data_ptr ? -1 : 1);
}

#ifndef __unused
#define __unused
#endif /* __unused */
RB_GENERATE_STATIC(tree, block, entry, tree_cmp)

static struct block *
find_block(struct tree *tree, uintptr_t data_ptr)
{
	struct block key, *block;

	key.data_ptr = data_ptr;
	block = RB_FIND(tree, tree, &key);

	return (block);
}

static int
extend_file(struct xm_allocator *allocator)
{
	size_t oldsize, newsize;

	oldsize = bitstr_size(allocator->file_bytes / XM_PAGE_SIZE);
	allocator->file_bytes = allocator->file_bytes > XM_GROW_SIZE ?
	    allocator->file_bytes + XM_GROW_SIZE :
	    allocator->file_bytes * 2;
	newsize = bitstr_size(allocator->file_bytes / XM_PAGE_SIZE);

	if (ftruncate(allocator->fd, (off_t)allocator->file_bytes)) {
		perror("ftruncate");
		return (1);
	}
	if ((allocator->pages = realloc(allocator->pages,
	    newsize * sizeof(bitstr_t))) == NULL) {
		perror("realloc");
		return (1);
	}
	memset(allocator->pages + oldsize, 0,
	    (newsize - oldsize) * sizeof(bitstr_t));
	return (0);
}

static uintptr_t
find_pages(struct xm_allocator *allocator, size_t n_pages)
{
	int i, n_free, n_total, offset, start;

	assert(n_pages > 0);

	n_total = (int)(allocator->file_bytes / XM_PAGE_SIZE);
	bit_ffc(allocator->pages, n_total, &start);
	if (start == -1)
		return (XM_NULL_PTR);
	for (i = start, n_free = 0; i < n_total; i++) {
		if (bit_test(allocator->pages, i))
			n_free = 0;
		else
			n_free++;
		if (n_free == (int)n_pages) {
			offset = i + 1 - n_free;
			bit_nset(allocator->pages, offset, i);
			return ((uintptr_t)offset * XM_PAGE_SIZE);
		}
	}

	return (XM_NULL_PTR);
}

static uintptr_t
allocate_pages(struct xm_allocator *allocator, size_t size_bytes)
{
	size_t n_pages;
	uintptr_t ptr;

	if (size_bytes == 0)
		return (XM_NULL_PTR);

	n_pages = (size_bytes + XM_PAGE_SIZE - 1) / XM_PAGE_SIZE;

	while ((ptr = find_pages(allocator, n_pages)) == XM_NULL_PTR)
		if (extend_file(allocator))
			return (XM_NULL_PTR);

	return (ptr);
}

struct xm_allocator *
xm_allocator_create(const char *path)
{
	struct xm_allocator *allocator;

	if ((allocator = calloc(1, sizeof(*allocator))) == NULL) {
		perror("malloc");
		return (NULL);
	}

	if (path) {
		if ((allocator->fd = open(path, O_CREAT|O_RDWR,
		    S_IRUSR|S_IWUSR)) == -1) {
			perror("open");
			free(allocator);
			return (NULL);
		}

		allocator->file_bytes = XM_PAGE_SIZE;
		allocator->pages = bit_alloc(1);

		if (ftruncate(allocator->fd, (off_t)allocator->file_bytes)) {
			perror("ftruncate");
			if (close(allocator->fd))
				perror("close");
			free(allocator);
			return (NULL);
		}
		if ((allocator->path = strdup(path)) == NULL) {
			perror("malloc");
			if (close(allocator->fd))
				perror("close");
			free(allocator);
			return (NULL);
		}
	}

	if (pthread_mutex_init(&allocator->mutex, NULL)) {
		perror("pthread_mutex_init");
		if (close(allocator->fd))
			perror("close");
		free(allocator->path);
		free(allocator);
		return (NULL);
	}
	RB_INIT(&allocator->blocks);
	return (allocator);
}

const char *
xm_allocator_get_path(struct xm_allocator *allocator)
{
	return (allocator->path);
}

uintptr_t
xm_allocator_allocate(struct xm_allocator *allocator, size_t size_bytes)
{
	struct block *block;
	void *data;

	if ((block = calloc(1, sizeof(*block))) == NULL) {
		perror("malloc");
		return (XM_NULL_PTR);
	}

	if (pthread_mutex_lock(&allocator->mutex))
		perror("pthread_mutex_lock");

	if (allocator->path) {
		if ((block->data_ptr = allocate_pages(allocator,
		    size_bytes)) == XM_NULL_PTR)
			goto fail;
	} else {
		if ((data = malloc(size_bytes)) == NULL) {
			perror("malloc");
			goto fail;
		}
		block->data_ptr = (uintptr_t)data;
	}

	block->size_bytes = size_bytes;
	RB_INSERT(tree, &allocator->blocks, block);

	if (pthread_mutex_unlock(&allocator->mutex))
		perror("pthread_mutex_unlock");
	return (block->data_ptr);
fail:
	if (pthread_mutex_unlock(&allocator->mutex))
		perror("pthread_mutex_unlock");
	free(block);
	return (XM_NULL_PTR);
}

void
xm_allocator_memset(struct xm_allocator *allocator, uintptr_t data_ptr,
    unsigned char c, size_t size_bytes)
{
	size_t write_bytes;
	ssize_t written;
	off_t offset;
	unsigned char buf[65536];

	assert(data_ptr != XM_NULL_PTR);

	if (allocator->path == NULL) {
		memset((void *)data_ptr, c, size_bytes);
		return;
	}

	memset(buf, c, sizeof buf);
	write_bytes = 0;
	for (offset = (off_t)data_ptr;
	     write_bytes + sizeof buf < size_bytes;
	     offset += sizeof buf) {
		written = pwrite(allocator->fd, buf, sizeof buf, offset);
		if (written != (ssize_t)(sizeof buf)) {
			perror("pwrite");
			abort();
		}
		write_bytes += sizeof buf;
	}
	written = pwrite(allocator->fd, buf, size_bytes - write_bytes, offset);
	if (written != (ssize_t)(size_bytes - write_bytes)) {
		perror("pwrite");
		abort();
	}
}

void
xm_allocator_read(struct xm_allocator *allocator, uintptr_t data_ptr,
    void *mem, size_t size_bytes)
{
	ssize_t read_bytes;
	off_t offset;

	assert(data_ptr != XM_NULL_PTR);

	if (allocator->path == NULL) {
		memcpy(mem, (const void *)data_ptr, size_bytes);
		return;
	}

	offset = (off_t)data_ptr;
	read_bytes = pread(allocator->fd, mem, size_bytes, offset);

	if (read_bytes != (ssize_t)size_bytes) {
		perror("pread");
		abort();
	}
}

void
xm_allocator_write(struct xm_allocator *allocator, uintptr_t data_ptr,
    const void *mem, size_t size_bytes)
{
	ssize_t write_bytes;
	off_t offset;

	assert(data_ptr != XM_NULL_PTR);

	if (allocator->path == NULL) {
		memcpy((void *)data_ptr, mem, size_bytes);
		return;
	}

	offset = (off_t)data_ptr;
	write_bytes = pwrite(allocator->fd, mem, size_bytes, offset);

	if (write_bytes != (ssize_t)size_bytes) {
		perror("pwrite");
		abort();
	}
}

void
xm_allocator_deallocate(struct xm_allocator *allocator, uintptr_t data_ptr)
{
	struct block *block;

	if (data_ptr == XM_NULL_PTR)
		return;
	if (pthread_mutex_lock(&allocator->mutex))
		perror("pthread_mutex_lock");

	block = find_block(&allocator->blocks, data_ptr);
	assert(block);

	RB_REMOVE(tree, &allocator->blocks, block);

	if (allocator->path) {
		int start, count;
		assert(data_ptr % XM_PAGE_SIZE == 0);
		start = (int)(data_ptr / XM_PAGE_SIZE);
		count = (int)((block->size_bytes - 1) / XM_PAGE_SIZE);
		bit_nclear(allocator->pages, start, start + count);
	} else {
		free((void *)data_ptr);
	}
	free(block);

	if (pthread_mutex_unlock(&allocator->mutex))
		perror("pthread_mutex_unlock");
}

void
xm_allocator_destroy(struct xm_allocator *allocator)
{
	struct block *block, *next;

	if (allocator) {
		RB_FOREACH_SAFE(block, tree, &allocator->blocks, next) {
			xm_allocator_deallocate(allocator, block->data_ptr);
		}
		if (allocator->path) {
			if (close(allocator->fd))
				perror("close");
			if (unlink(allocator->path))
				perror("unlink");
			free(allocator->path);
		}
		if (pthread_mutex_destroy(&allocator->mutex))
			perror("pthread_mutex_destroy");
		free(allocator->pages);
		free(allocator);
	}
}
