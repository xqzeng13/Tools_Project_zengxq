import numpy as np
import os
import glob
import nibabel as nib
import SimpleITK as sitk



def conv3d_data2(data,k_size):#k_size[c,h,w][3,1,1]
    padd_c = k_size[0]//2
    padd_h = k_size[1]//2
    padd_w = k_size[2]//2
    x_pad = np.pad(data,((padd_c,padd_c),(padd_h,padd_h),(padd_w,padd_w)))#(130, 448, 448)
    c_num = data.shape[0]
    h_num = data.shape[1]
    w_num = data.shape[2]
    temp_c = np.array([x_pad[i:i+k_size[0],:,:] for i in range(c_num)])
    temp_h = np.array([temp_c[:,:,i:i+k_size[1],:] for i in range(h_num)])
    temp_w = np.array([temp_h]*k_size[2]).transpose([2,1,0,3,4,5])
    for i in range(1,k_size[2]):
        temp_w[:,:,i,:,:,:-i] = temp_w[:,:,i,:,:,i:]
    result = temp_w[:,:,:,:,:,:-(k_size[2])].transpose([0,1,5,3,4,2])
    return result

def generate_mask3d(img_depth,img_height,img_width,radius,center_z,center_y,center_x):
    y = np.array(list(range(img_height))).reshape([1,img_height,1])
    x = np.array(list(range(img_width))).reshape([1,1,img_width])
    z = np.array(list(range(img_depth))).reshape([img_depth,1,1])
    mask = (x-center_x)**2+(y-center_y)**2+(z-center_z)**2<=radius**2
    return mask

#TODO 膨胀
def dilate_circle3d(img,K_size):
    data = conv3d_data2(img>0,K_size)
    r = int(np.min(K_size)//2)
    center_z = int(K_size[0]//2)
    center_y = int(K_size[1]//2)
    center_x = int(K_size[2]//2)
    K = generate_mask3d(K_size[0],K_size[1],K_size[2],r,center_z,center_y,center_x)*1
    result = np.einsum("abcdef,def->abc",data,K)
    result = np.array(result>0)*1
    return result
def dilate_rect3d(img,K_size):
    data = conv3d_data2(img>0,K_size)
    K = np.ones(K_size)
    result = np.einsum("abcdef,def->abc",data,K)
    result = (result>0)*1
    return result

#TODO 腐蚀
def erode_rect3d(img,K_size):
    data = conv3d_data2(img>0,K_size)
    K = np.ones(K_size)
    summ = K.shape[0]*K.shape[1]*K.shape[2]
    result = np.einsum("abcdef,def->abc",data,K)
    result = (result==summ)*1
    return result
def erode_circle3d(img,K_size):
    data = conv3d_data2(img>0,K_size)
    r = int(np.min(K_size)//2)
    center_z = int(K_size[0]//2)
    center_y = int(K_size[1]//2)
    center_x = int(K_size[2]//2)
    K = generate_mask3d(K_size[0],K_size[1],K_size[2],r,center_z,center_y,center_x)*1
    summ =np.sum(K)
    result = np.einsum("abcdef,def->abc",data,K)
    result = (result==summ)*1
    return result

if __name__ == '__main__':
    pred=r'D:\Work\Tools_Project_zengxq\output\\'
    output_result=r'D:\Work\Tools_Project_zengxq\output_result\\'
    predlist=sorted(glob.glob(os.path.join(pred, '*.nii.gz')))
    #TODO-----------------------------------------
    k1_size=[3,3,3]##处理区域
    k2_size=[2,2,2]##处理区域

    for i in range(len(predlist)):
        pred=predlist[i]
        pred_arr_nib=nib.load(pred)
        pred_arr = pred_arr_nib.get_fdata()
        pred_arr=pred_arr.swapaxes(0, 2)
        # out_arr=erode_circle3d(pred_arr,k_size)#圆形腐蚀
        # out_arr=erode_rect3d(pred_arr,k_size)#方形腐蚀
        out_arr_dilate=dilate_rect3d(pred_arr,k2_size)#膨胀dilate_rect3d

        out_arr_erode = erode_rect3d(pred_arr, k2_size)  # 圆形腐蚀

        dataimage = r'D:\Work\Datasets\GoldNormaldatas20\label\Normal074-MRA.nii.gz'
        # TODO 获取图像坐标SPCING信息
        data1 = sitk.ReadImage(dataimage)
        # out_arr_dilate = out_arr_dilate.swapaxes(0, 2)
        pred_img1 = sitk.GetImageFromArray((np.array(out_arr_dilate, dtype='uint8')))
        pred_img1.SetDirection(data1.GetDirection())  ###########图像方向不变
        pred_img1.SetOrigin(data1.GetOrigin())  ###########图像原点不变
        pred_img1.SetSpacing((data1.GetSpacing()[0], data1.GetSpacing()[1], data1.GetSpacing()[2]))
        sitk.WriteImage(pred_img1, os.path.join(r'D:\Work\Tools_Project_zengxq\output\result\\',
                                               'patch_result-dilate_2' + '.nii.gz'))

        # out_arr_erode = out_arr_erode.swapaxes(0, 2)
        pred_img2 = sitk.GetImageFromArray((np.array(out_arr_erode, dtype='uint8')))
        pred_img2.SetDirection(data1.GetDirection())  ###########图像方向不变
        pred_img2.SetOrigin(data1.GetOrigin())  ###########图像原点不变
        pred_img2.SetSpacing((data1.GetSpacing()[0], data1.GetSpacing()[1], data1.GetSpacing()[2]))
        sitk.WriteImage(pred_img2, os.path.join(r'D:\Work\Tools_Project_zengxq\output\result\\',
                                               'patch_result-ps_2' + '.nii.gz'))



        print("over")
