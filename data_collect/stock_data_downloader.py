# -*- coding:utf8 -*-
import tushare as ts
import pandas as pd
import mysql.connector
import os
from sqlalchemy import create_engine

import auto_run_daily_v2

# stock_code_dict={
#     u'陶瓷行业':'taocihangye',
#     u'房地产':'fangdichan',
#     u'家电行业':'jiadianhangye',
#     u'建筑建材':'jianzhujiancai',
#     u'开发区':'kaifaqu',
#     u'供水供气':'gongshuigongqi',
#     u'印刷包装':'yingshuabaozhuang',
#     u'石油行业':'shiyouhangye',
#     u'塑料制品':'suliaozhiping',
#     u'电力行业':'dianlihangye',
#     u'传媒娱乐':'chuanmeiyule',
#     u'电器行业':'dianqihangye',
#     u'飞机制造':'feijizhizhao',
#     u'家具行业':'jiadianhangye',
#     u'发电设备':'fadianshebei',
#     u'酿酒行业':'niangjiuhangye',
#     u'钢铁行业':'gangtiehangye',
#     u'综合行业':'zonghehangye',
#     u'纺织机械':'fangzhijixie',
#     u'医疗器械':'yiliaoqixie',
#     u'电子器件':'dianziqijian',
#     u'化工行业':'huagonghangye',
#     u'船舶制造':'chuanbozhizhao',
#     u'摩托车':'motuoche',
#     u'酒店旅游':'jiudianlvyou',
#     u'电子信息':'dianzixinxi',
#     u'玻璃行业':'bolihangye',
#     u'商业百货':'shangyebiahuo',
#     u'造纸行业':'zhaozhihangye',
#     u'其它行业':'qitahangye',
#     u'仪器仪表':'yiqiyibiao',
#     u'生物制药':'shengwuzhiyao',
#     u'食品行业':'shipinghangye',
#     u'有色金属':'yousejinshu',
#     u'农林牧渔':'nonglinmuyu',
#     u'次新股':'cixingu',
#     u'机械行业':'jixiehangye',
#     u'水泥行业':'shuinihangye',
#     u'化纤行业':'huaqianhangye',
#     u'公路桥梁':'gongluqiaoliang',
#     u'环保行业':'huanbaohangye',
#     u'汽车制造':'qichezhizhao',
#     u'物资外贸':'wuziwaimao',
#     u'煤炭行业':'meitanhangye',
#     u'服装鞋类':'fuzhuangxielei',
#     u'农药化肥':'nongyaohuafei',
#     u'纺织行业':'fangzhihangye',
#     u'金融行业':'jinronghangye'
#     }

stock_code_dict_sw = {
    u'化学制药': 'huaxuezhiyao',
    u'保险': 'baoxian',
    u'运输设备': 'yunshushebei',
    u'白色家电': 'baisejiadian',
    u'种植业': 'zhongzhiye',
    u'航空运输': 'hangkongyunshu',
    u'酒店': 'jiudian',
    u'其他电子': 'qitadianzi',
    u'元件': 'yuanjian',
    u'饮料制造': 'yinliaozhizao',
    u'通信': 'tongxin',
    u'其他建材': 'qitajiancai',
    u'专业工程': 'zhuanyegongcheng',
    u'通信运营': 'tongxinyunying',
    u'动物保健': 'dongwubaojian',
    u'工业金属': 'gongyejinshu',
    u'医疗服务': 'yiliaofuwu',
    u'园区开发': 'yuanqukaifa',
    u'纺织服装': 'fangzhifuzhuang',
    u'家用电器': 'jiayongdianqi',
    u'医药商业': 'yiyaoshangye',
    u'高低压设备': 'gaodiyashebei',
    u'燃气': 'ranqi',
    u'汽车服务': 'qichefuwu',
    u'中药': 'zhongyao',
    u'光学光电子': 'guangxueguangdianzi',
    u'电子制造': 'dianzizhizao',
    u'铁路运输': 'tieluyunshu',
    u'互联网传媒': 'hulianwangchuanmei',
    u'综合': 'zonghe',
    u'医药生物': 'yiyaoshengwu',
    u'计算机应用': 'jisuanjiyingyong',
    u'化学制品': 'huaxuezhipin',
    u'航天装备': 'hangtianzhuanbei',
    u'饲料': 'siliao',
    u'电力': 'dianli',
    u'金属非金属新材料': 'jinshufeijinshuxincailiao',
    u'化学原料': 'huanxueyuanliao',
    u'机场': 'jichang',
    u'景点': 'jingdian',
    u'电气自动化设备': 'dianqizidonghuashebei',
    u'传媒': 'chuanmei',
    u'房地产开发': 'fangdichankaifa',
    u'贸易': 'maoyi',
    #u'化工': 'huagong',
    u'公用事业': 'gongyongshiye',
    u'建筑装饰': 'jianzhuzhuangshi',
    u'半导体': 'bandaoti',
    u'通用机械': 'tongyongjixie',
    u'装修装饰': 'zhuangxiuzhuangshi',
    u'生物制品': 'shengwuzhipin',
    u'文化传媒': 'wenhuachuanmei',
    u'玻璃制造': 'bolizhizao',
    u'汽车整车': 'qichezhengche',
    u'商业贸易': 'shangyemaoyi',
    u'房屋建设': 'fangwujianshe',
    u'国防军工': 'guofangjungong',
    u'渔业': 'yuye',
    u'食品加工': 'shipinjiagong',
    u'其他休闲服务': 'qitaxiuxianfuwu',
    u'水泥制造': 'shuinizhizao',
    u'休闲服务': 'xiuxianfuwu',
    u'电源设备': 'dianyuanshebei',
    u'商业物业经营': 'shangyewuyejingying',
    u'非银金融': 'feiyinjinrong',
    u'通信设备': 'tongxinshebei',
    u'港口': 'gangkou',
    u'农业综合': 'nongyezonghe',
    u'汽车': 'qiche',
    u'化学纤维': 'huaxuexianwei',
    u'船舶制造': 'chuanbozhizao',
    u'机械设备': 'jixieshebei',
    u'稀有金属': 'xiyoujinshu',
    u'基础建设': 'jichujianshe',
    u'汽车零部件': 'qichelingbujian',
    u'公交': 'gongjiao',
    u'其他采掘': 'qitacaijue',
    u'食品饮料': 'shipinyinliao',
    u'建筑材料': 'jianzhucailiao',
    u'纺织制造': 'fangzhizhizao',
    u'旅游综合': 'lvyouzonghe',
    u'轻工制造': 'qingongzhizao',
    u'医疗器械': 'yiliaoqixie',
    u'交通运输': 'jiaotongyunshu',
    u'一般零售': 'yibanlingshou',
    u'计算机设备': 'jisuanjishebei',
    u'其他交运设备': 'qitajiaoyunshebei',
    u'电机': 'dianji',
    u'林业': 'linye',
    #u'计算机': 'jisuanji',
    u'黄金': 'huanjin',
    u'农产品加工': 'nongchanpinjiagong',
    u'造纸': 'zaozhi',
    u'包装印刷': 'baozhuangyinshua',
    u'水务': 'shuiwu',
    u'家用轻工': 'jiayongqingong',
    u'专用设备': 'zhuanyongshebei',
    u'视听器材': 'shitingqicai',
    u'石油开采': 'shiyoukaicai',
    u'畜禽养殖': 'chuqinyangzhi',
    u'高速公路': 'gaosugonglu',
    u'其他轻工制造': 'qitaqinggongzhizao',
    u'有色金属': 'yousejinshu',
    u'煤炭开采': 'meitankaicai',
    u'电子': 'dianzi',
    u'园林工程': 'yuanlinggongcheng',
    u'环保工程及服务': 'huanbaogongchengjifuwu',
    u'银行': 'yinhang',
    u'地面兵装': 'dimianbinzhuang',
    u'专业零售': 'zhuanyelingshou',
    u'塑料': 'suliao',
    u'营销传播': 'yingxiaochuanbo',
    u'钢铁': 'gangtie',
    u'航运': 'hangyun',
    u'多元金融': 'duoyuanjinrong',
    u'电气设备': 'dianqishebei',
    u'房地产': 'fangdichan',
    u'橡胶': 'xiangjiao',
    u'航空装备': 'hangkongzhuangbei',
    u'农林牧渔': 'nonglingmuyu',
    u'采掘': 'caijue',
    u'金属制品': 'jinshuzhipin',
    u'服装家纺': 'fuzhuanjiafang',
    u'仪器仪表': 'yiqiyibiao',
    u'证券': 'zhengquan',
    u'餐饮': 'canyin',
    u'石油化工': 'shiyouhuagong',
    u'物流': 'wuliu',
    u'采掘服务': 'caijuefuwu'
}


def get_stock_code_by_industry(industry=u'综合行业', standard='sw'):
    # all_stock=ts.get_industry_classified()#get data from sw or sina
    if standard == 'sw':
        all_stock = pd.read_csv('../data/resource/all_industry_stock_from_sw.csv',
                                encoding='utf8')  # we read data from csv file that was downloaded previous
    else:
        all_stock = pd.read_csv('../data/resource/all_industry_stock.csv', encoding='utf8')
    return all_stock[all_stock['c_name'] == industry]


def filter_st(stocks):
    return stocks[
        ~(stocks.name.str.startswith('ST') | stocks.name.str.startswith('*ST') | stocks.name.str.startswith(u'退'))]


def download_stock_data(industry=u'综合行业', start_date='', end_date=None, k_type='D',root_dir='../data/industry_sw/'):
    stock_codes = get_stock_code_by_industry(industry)
    stock_codes = filter_st(stock_codes)

    dir = root_dir + stock_code_dict_sw[industry]
    if not os.path.exists(dir):
        os.makedirs(dir)
    i = 0
    size = stock_codes.shape[0]
    for code in stock_codes['code']:
        i += 1
        code = str(code)
        print(code)
        data = ts.get_k_data(code, start=start_date, end=end_date, ktype=k_type)
        if data is None or data.empty:
            continue
        data = data[data.volume > 1e-3]
        data.to_csv(dir + '/' + code + '.csv')

        print('%d/%d' % (i, size))


def filter_data(industry=''):
    # dir = '../data/industry/' + industry
    dir = '../data/nasdaq/all/test' + industry
    if not os.path.exists(dir):
        return
    for file in os.listdir(dir):
        file = dir + '/' + file
        if os.path.isfile(file):
            data = pd.read_csv(file)
            data = data[data.volume > 1e-3]
            data.to_csv(file)
            print(file)
def update_hushen_stock_info():
    empty_table('hushen_stock_info')

    df=ts.get_stock_basics()
    mysql_con=create_engine('mysql+pymysql://root:yangxh@localhost:3306/quant_bee?charset=utf8')
    df.rename(columns={'pe':'price_earning_ratio', 'totalAssets':'total_assets','liquidAssets':'liquid_assets','fixedAssets':'fixed_assets'
                       ,'reservedPerShare':'reserved_per_share','timeToMarket':'time_to_market','perundp':'per_undp'}, inplace = True)
    df=df[df['time_to_market']!=0]#过滤退市
    df['time_to_market'].apply(format_time_to_market)
    df.to_sql('hushen_stock_info',mysql_con,if_exists='append')
def format_time_to_market(x):
    x=str(x)
    return x[0:4]+'-'+x[4:6]+'-'+x[6:]

def empty_table(table=''):
    db = mysql.connector.connect(user='root', password='yangxh', database='quant_bee', use_unicode=True)
    cursor = db.cursor()
    sql = 'delete from '+table
    auto_run_daily_v2.db_update(db, cursor, sql)
    sql = 'alter table '+table+' auto_increment=0'
    auto_run_daily_v2.db_update(db, cursor, sql)
    db.close()

def update_hushen_stock_list_by_industry():
    df=ts.get_industry_classified(standard='sw')
    df.to_csv('../data/resource/all_industry_stock_from_sw.csv',encoding='utf8')

def update_hushen_stock_list_by_concept():
    empty_table('hushen_stock_concept')

    df=ts.get_concept_classified()

    df.to_csv('../data/resource/all_industry_stock_from_sw_by_concept.csv',encoding='utf8')

    df=df[['code','c_name']]
    df.rename(columns={'c_name':'concept'},inplace=True)
    mysql_con = create_engine('mysql+pymysql://root:yangxh@localhost:3306/quant_bee?charset=utf8')
    df.to_sql('hushen_stock_concept', mysql_con, if_exists='append',index=False)


def update_resource(update_stock_info=False,update_concept=False,update_industry=False):
    if update_stock_info:
        update_hushen_stock_info()
    if update_concept:
        update_hushen_stock_list_by_concept()
    if update_industry:
        update_hushen_stock_list_by_industry()

if __name__ == '__main__':
    # minutes=['5','15','30','60']
    # for m in minutes:
    #     i = 0
    #     size = len(stock_code_dict_sw)
    #     for code, _ in stock_code_dict_sw.items():
    #         print(code)
    #         i += 1
    #         print('%d/%d' % (i, size))
    #         download_stock_data(industry=code, start_date='2017-01-01',k_type=m,root_dir='../data/industry_sw_m'+m+'/')
    #化工,计算机
    # download_stock_data(industry=u'计算机', start_date='2010-01-01')
    # filter_data()
    update_resource(update_stock_info=False,update_concept=True)
    # conn=ts.get_apis()
    # print ts.bar(code='000001',conn=conn,start_date='2018-03-11',end_date='2018-03-12',freq='D')