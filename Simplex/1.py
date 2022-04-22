seasons = ['小狗', '小猫', '便便', '猫粮']

print('=======当默认start为0时，结果如下：========')
for index,value in enumerate(seasons):#默认start从0开始枚举

    print(f'索引为{index}时: 值为 {value}')

print('=======当默认start为1时，结果如下：========')
for index, value in enumerate(seasons,1):  # 默认start从1开始枚举

    print(f'索引为{index}时: 值为 {value}')



    # a = list(enumerate(seasons))
    # print('这是默认start从0开始的结果：\n', a)
    #
    # b = list(enumerate(seasons, start=1))
    # print('这是默认start从1开始的结果：\n', b)
