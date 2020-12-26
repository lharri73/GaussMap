#include <cstdlib>

class gaussMap{
    private:
        int* array;
        size_t width;
        size_t height;

    public:
        const int* Array() const;
        size_t Width() const;
        size_t Height() const;
};
