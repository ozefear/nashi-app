import XCTest
import CoreGraphics
import Foundation
@testable import NashiCore

final class ImageBackgroundRemovalTests:XCTestCase {
    private func fixture()throws->CGImage {
        var pixels=[UInt8](repeating:255,count:7*7*4)
        for y in 2...4 { for x in 2...4 { let offset=(y*7+x)*4;pixels[offset]=255;pixels[offset+1]=0;pixels[offset+2]=0 } }
        let enclosed=(3*7+3)*4;pixels[enclosed]=255;pixels[enclosed+1]=255;pixels[enclosed+2]=255
        let provider=try XCTUnwrap(CGDataProvider(data:Data(pixels) as CFData))
        return try XCTUnwrap(CGImage(width:7,height:7,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:28,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedLast.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent))
    }
    private func pixels(_ image:CGImage)throws->[UInt8] {
        var result=[UInt8](repeating:0,count:image.width*image.height*4)
        let rendered=result.withUnsafeMutableBytes { bytes->Bool in
            guard let context=CGContext(data:bytes.baseAddress,width:image.width,height:image.height,bitsPerComponent:8,bytesPerRow:image.width*4,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else{return false}
            context.draw(image,in:CGRect(x:0,y:0,width:image.width,height:image.height));return true
        }
        XCTAssertTrue(rendered);return result
    }
    func testOnlyEdgeConnectedBackgroundBecomesTransparent()throws {
        let output=try ImageEngine.removeBackground(fixture(),options:.init(),token:.init()),bytes=try pixels(output)
        XCTAssertEqual(bytes[3],0);XCTAssertEqual(bytes[(2*7+2)*4+3],255);XCTAssertEqual(bytes[(3*7+3)*4+3],255)
    }
    func testCancellationAndCornerConsensusAreEnforced()throws {
        let token=CancellationToken();token.cancel()
        XCTAssertThrowsError(try ImageEngine.removeBackground(fixture(),options:.init(),token:token))
        var varied=[UInt8](repeating:0,count:4*4*4);for index in 0..<16 { varied[index*4+3]=255 }
        let colors:[(UInt8,UInt8,UInt8)]=[(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        for (index,color) in zip([0,3,12,15],colors) { varied[index*4]=color.0;varied[index*4+1]=color.1;varied[index*4+2]=color.2 }
        let provider=try XCTUnwrap(CGDataProvider(data:Data(varied) as CFData));let image=try XCTUnwrap(CGImage(width:4,height:4,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:16,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.premultipliedLast.rawValue),provider:provider,decode:nil,shouldInterpolate:false,intent:.defaultIntent))
        XCTAssertThrowsError(try ImageEngine.removeBackground(image,options:.init(),token:.init()))
    }
}
