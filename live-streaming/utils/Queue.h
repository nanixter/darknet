#ifndef LIVESTREAMDETECTOR_QUEUE_H
#define LIVESTREAMDETECTOR_QUEUE_H
#include <queue>
#include <mutex>
#include <condition_variable>

namespace LiveStreamDetector {

	template <class T>
	class MutexQueue {
	public:
		void push_back(T &elem) {
			std::lock_guard<std::mutex> lock(this->mutex);
			this->queue.push(elem);
		}

		void push_back(std::vector<T> &elems) {
			std::lock_guard<std::mutex> lock(this->mutex);
			for (auto elemIterator = elems.begin(); elemIterator != elems.end(); elemIterator++) {
				this->queue.push(*elemIterator);
			}
		}

		// pops 1 element
		bool pop_front(T &elem) {
			if (this->queue.empty())
				return false;

			std::unique_lock<std::mutex> lock(this->mutex);
			if(!this->queue.empty()) {
				elem = (this->queue.front());
				this->queue.pop();
				return true;
			}
			return false;
		}

		// Pops upto N elements
		bool pop_front(std::vector<T> &elems, int &numElems) {
			if (this->queue.empty())
				return false;

			std::unique_lock<std::mutex> lock(this->mutex);
			int numPopped = 0;
			while( !this->queue.empty() && (numPopped < numElems) ) {
				elems.insert(elems.end(), this->queue.front());
				this->queue.pop();
				numPopped++;
			}
			numElems = numPopped;
			return true;
		}

	private:
		std::queue<T> queue;
		std::mutex mutex;

	}; // class DetectionQueue

} // namespace

#endif // QUEUE_H
