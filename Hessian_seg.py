# import SimpleITK as itk
# import itk
# import SimpleITK as sitk
import itk
import nibabel as nib
import numpy as np
from skimage.feature import canny
import SimpleITK as sitk
import pandas as pd
def vessleSegment(niipath):
    sigma_minimum = 0.2
    sigma_maximum = 3.
    number_of_sigma_steps = 8
    lowerThreshold = 40
    output_image = 'vessel.mha'
    input_image = itk.imread(niipath)
    # print("is ok ")
    # sitk.Symmtricsecond
    # 1.采用itk的多尺度hessian矩阵进行血管增强
    ImageType = type(input_image)
    Dimension = input_image.GetImageDimension()
    HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]


    objectness_filter = itk.HessianToObjectnessMeasureImageFilter[HessianImageType, ImageType].New()
    objectness_filter.SetBrightObject(True)
    objectness_filter.SetScaleObjectnessMeasure(True)
    objectness_filter.SetAlpha(0.5)
    objectness_filter.SetBeta(1.0)
    objectness_filter.SetGamma(5.0)
    multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[ImageType, HessianImageType, ImageType].New()

    multi_scale_filter.SetInput(input_image)

    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()
    multi_scale_filter.SetSigmaMinimum(sigma_minimum)
    multi_scale_filter.SetSigmaMaximum(sigma_maximum)
    multi_scale_filter.SetNumberOfSigmaSteps(number_of_sigma_steps)
    # itk.RescaleIntensityImageFilter[]
    result_01=r'D:\Work\Datasets\ReSamples\hessian_nii_val\hessian_result\\'+str(input_df.iloc[i].at['filename']).split('seg')[0]+'.nii.gz'
    itk.imwrite(multi_scale_filter.GetOutput(), result_01)
    img_arr = nib.load(result_01).get_fdata()
    ###归一化到0-1
    if img_arr.max()<=0:

        img_arr=np.zeros(img_arr.shape)##应该是0矩阵
    else:
        img_arr=img_arr/((img_arr.max())-(img_arr.min()))
    return img_arr
        # nii_file_nor=r'D:\Work\Datasets\ReSamples\hessian_nii\hessian_result\\'+str(input_df.iloc[i].at['filename']).split('seg')[0]+'.npy'
        # np.save( nii_file_nor,img_arr)

        # nii_file_nor=result_01
        # new_image = nib.Nifti1Image(img_arr, np.eye(4))####保存为nii格式
        # nib.save(new_image, nii_file_nor)
        # nib.save(img_arr, nii_file)
    # ##todo 测试
    # img_arr = sitk.ReadImage(r'.\step1.nii.gz')
    # img_arr = sitk.ReadImage(r'.\step1.nii.gz')
    # img_arr[img_arr > 0.001] = 1
    # img_arr[img_arr < 0.001] = 0
    # image = img_arr.swapaxes(0, 2)
    # import numpy as np
    # import nibabel as nib
    # new_image = nib.Nifti1Image(image, np.eye(4))
    # nib.save(new_image, 'nifti.nii.gz')


if __name__ == '__main__':
    print("Start Hessian_edge detection!")
    csvname = r'D:\Work\Datasets\ReSamples\train.csv'  # train_only_vessel_new.csv  train_only_vessel_new+random.csv D:\Work\Datasets\new_samples\train_only_vessel_new.csv
    save_csv=r'D:\Work\Datasets\ReSamples\train_edge_hessian.csv'

    pathname = r'D:\Work\Datasets\ReSamples\train_val\\'
    hessianname='D:\Work\Datasets\ReSamples\hessian_nii\hessian_result\\'
    # edgesname1='D:\Work\Datasets\ReSamples\edge1\\'

    input_df = pd.read_csv(csvname)
    print("\n sum of train patches is :", input_df.shape[0])
    pathnamelist=[]
    filenamelist = []
    hessiannametlist=[]
    hessianfilenamelist=[]
    niifillenamelist=[]
    j=0
    for i in range(input_df.shape[0]):
        labelname = pathname + input_df.iloc[i].at['filename']
        dataname = pathname + str(input_df.iloc[i].at['filename']).split('seg')[0] + '0.npy'
        img_arr=np.load(dataname)
        # img = sitk.GetImageFromArray(img_arr)
        new_image = nib.Nifti1Image(img_arr, np.eye(4))
        nii_file=r'D:\Work\Datasets\ReSamples\hessian_nii\\'+str(input_df.iloc[i].at['filename']).split('seg')[0]+'.nii.gz'
        nib.save(new_image, nii_file)
        # niifillenamelist.append(nii_file)
        hessian_out=vessleSegment(nii_file)
        # nii_file_nor=r'D:\Work\Datasets\ReSamples\hessian_nii\hessian_result\\'+str(input_df.iloc[i].at['filename']).split('seg')[0]+'.npy'
        # np.save( nii_file_nor,img_arr)
        hessianfilename =str(input_df.iloc[i].at['filename']).split('seg')[0]+'hessian.npy'

        hessianfilenamelist.append(hessianfilename)
        hessiannametlist.append(hessianname)
        filenamelist.append(input_df.iloc[i].at['filename'])
        pathnamelist.append(pathname)
        output_excel = {'pathname': [], 'filename': [], 'hessiansname': [], 'hessianfilename': []}

        output_excel['pathname'] = pathnamelist
        output_excel['filename'] = filenamelist
        output_excel['hessiansname'] = hessiannametlist
        output_excel['hessianfilename'] = hessianfilenamelist
        output = pd.DataFrame(output_excel)
        output.to_csv(save_csv, index=False)
        print('\r[ %d / %d]' % (i, input_df.shape[0]), end='')

        np.save(hessianname + hessianfilename, hessian_out)
