unsigned int num2x(const unsigned int num, const unsigned int num_col)
{
    return (int) (num/num_col);
}

unsigned int num2y(const unsigned int num, const unsigned int num_col)
{
    return num % num_col;
}
