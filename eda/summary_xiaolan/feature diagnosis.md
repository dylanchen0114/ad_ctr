

####Oneshot

| Feature name                | Mode       | Comments                 | Reject/ accept |
| :-------------------------- | :--------- | ------------------------ | -------------- |
| item_brand_id_oneshot_ratio | rolling    | train/ valid discrepancy | Reject         |
| shop_id_oneshot_ratio       | rolling    | train/valid discrepancy  | Reject         |
| item_id_oneshot_ratio       | rolling    | train/valid discrepancy  | Reject         |
| item_id_oneshot_ratio       | cumulative | Need further processing  | Accept         |
| shop_id_oneshot_ratio       | cumulative | No signal                | Reject         |
| item_brand_id_oneshot_ratio | cumulative | Need further processing  | Pending        |

_(rolling)_

![image-20180414183934241](/Users/Effy/Documents/Git/alimama_v2/eda/summary_xiaolan/image-20180414183934241.png)

![image-20180414184615884](/Users/Effy/Documents/Git/alimama_v2/eda/summary_xiaolan/image-20180414184615884.png)

![image-20180414232929316](/Users/Effy/Documents/Git/alimama_v2/eda/summary_xiaolan/image-20180414232929316.png)

_(cumulative)_

![image-20180414232517782](/Users/Effy/Documents/Git/alimama_v2/eda/summary_xiaolan/image-20180414232517782.png)

![image-20180414232547675](/Users/Effy/Documents/Git/alimama_v2/eda/summary_xiaolan/image-20180414232547675.png)

![image-20180414232608135](/Users/Effy/Documents/Git/alimama_v2/eda/summary_xiaolan/image-20180414232608135.png)



#### Inarow

| Feature name                  | Mode | Comments                | Reject/ accept |
| :---------------------------- | :--- | ----------------------- | -------------- |
| item_brand_id_inarow          | NA   | No signal               | Reject         |
| shop_id_inarow                | NA   | train/valid discrepancy | Reject         |
| item_category_1_inarow_inarow | NA   | train/valid discrepancy | Reject         |

![image-20180414233114089](/Users/Effy/Documents/Git/alimama_v2/eda/summary_xiaolan/image-20180414233114089.png)

![image-20180414233800258](/Users/Effy/Documents/Git/alimama_v2/eda/summary_xiaolan/image-20180414233800258.png)

![image-20180414234152127](/Users/Effy/Documents/Git/alimama_v2/eda/summary_xiaolan/image-20180414234152127.png)



#### Match-diff

| Feature name                                       | Mode   | Comments                                  | Reject/ accept |
| :------------------------------------------------- | :----- | ----------------------------------------- | -------------- |
| user_age_level_group_shop_id_item_price_level_diff | global | Strong signal; try rolling and cumulative | Accept         |
| user_id_group_shop_id_item_price_level_diff        | global | Strong signal; try rolling and cumulative | Accept         |
|                                                    |        |                                           |                |

![image-20180415000755201](/Users/Effy/Documents/Git/alimama_v2/eda/summary_xiaolan/image-20180415000755201.png)

![image-20180415001506920](/var/folders/7d/pr_fc_js2yl1_kp7rv3svqrh0000gn/T/abnerworks.Typora/image-20180415001506920.png)

