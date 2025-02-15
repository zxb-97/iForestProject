package thesis.project;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import static thesis.project.NpzHeaderParser.*;

public class NpzParser {
    private final ZipFile zipFile;

    public NpzParser(String npzFilePath) throws IOException {
        zipFile = new ZipFile(npzFilePath);
    }

    // Transform the file into an InputStream
    // All datasets are named "X.npy" and the respective set of labels is "y.npy"
    public InputStream getXInputStream() throws IOException {
        ZipEntry x = zipFile.getEntry("X.npy");
        return zipFile.getInputStream(x);
    }
    public InputStream getYInputStream() throws IOException {
        ZipEntry y = zipFile.getEntry("y.npy");
        return zipFile.getInputStream(y);
    }

    public double[][] parseX() throws IOException {
        // Get the input stream to extract the npy header
        InputStream isMeta = getXInputStream();
        // Parse the npy header
        Map<String,String> metadata = NpzHeaderParser.npzHeaderReader(isMeta);

        int bufferSize = metadata.get("descr").charAt(3) - '0'; // For example '<f8' , '<i2'
        char ndarrayType = metadata.get("descr").charAt(2);

        // magicNumber(6 bytes) + NumpyVersion(2 bytes) + lengthOfHeader(2 bytes) = 10
        int offset = 10 + headerLength;
        int rows = getShapeX(metadata)[0];
        int cols = getShapeX(metadata)[1];
        double[][] data = new double[rows][cols];

        // Get input stream again since after parsing metadata the stream gets closed
        InputStream is = getXInputStream();

        try(BufferedInputStream bis = new BufferedInputStream(is)){


            byte[] buffer = new byte[bufferSize];
            // All files are in little endian, while java expects big endian
            // therefore we must explicitly specify that the byte stream is in little endian
            ByteBuffer byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);
            // Skip the header
            bis.skipNBytes(offset);

            // All X.npy files are stored in Fortran order
            for(int i = 0; i < cols; i++){
                for(int j = 0; j < rows; j++){
                    // Read the stream into the buffer
                    if(bis.read(buffer) != bufferSize){
                        throw new IOException("failed to read in row: " + j + " col: " + i);
                    }
                    // Once the buffer is read, the pointer point at the last position
                    // therefore we must re-initialize it
                    byteBuffer.rewind();

                    if(ndarrayType=='f'){
                        data[j][i] = byteBuffer.getDouble();
                    } else if (ndarrayType=='i' && bufferSize==8) {
                        data[j][i] = (double) byteBuffer.getLong();
                    } else if (ndarrayType=='i' && bufferSize==4) {
                        data[j][i] = (double) byteBuffer.getInt();
                    } else if (ndarrayType=='i' && bufferSize==2) {
                        data[j][i] = (double) byteBuffer.getShort();
                    }else throw new IOException("ndarray type of X not supported");
                }
            }
            return data;
        }
    }

    public long[] parseY() throws IOException {
        // Get the input stream to extract the npy header
        InputStream isMeta = getYInputStream();
        // Parse the npy header
        Map<String,String> metadata = NpzHeaderParser.npzHeaderReader(isMeta);

        int bufferSize = metadata.get("descr").charAt(3) - '0';  // For example '<f8' , '<i2'
        char ndarrayType = metadata.get("descr").charAt(2);
        // magicNumber(6 bytes) + NumpyVersion(2 bytes) + lengthOfHeader(2 bytes) = 10
        int offset = 10 + headerLength;

        int rows = getShapeY(metadata);
        long[] dataY = new long[rows];

        // Get input stream again since after parsing metadata the stream gets closed
        InputStream is = getYInputStream();

        try(BufferedInputStream bis = new BufferedInputStream(is)){
            byte[] buffer = new byte[bufferSize];
            // All files are in little endian, while java expects big endian
            // therefore we must explicitly specify that the byte stream is in little endian
            ByteBuffer byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);
            // Skip header
            bis.skipNBytes(offset);

            // All y.npy files are in c-order
            for(int i = 0; i < rows; i++){
                // Read the stream into the buffer
                if(bis.read(buffer) != bufferSize){
                    throw new IOException("failed to read at row: " + i);
                }
                // Once the buffer is read, the pointer point at the last position
                // therefore we must re-initialize it
                byteBuffer.rewind();

                if(ndarrayType=='i' && bufferSize == 8){
                    dataY[i] = byteBuffer.getLong();
                } else if (ndarrayType=='i' && bufferSize == 4) {
                    dataY[i] = byteBuffer.getInt();
                } else if (ndarrayType=='i' && bufferSize == 2) {
                    dataY[i] = byteBuffer.getShort();
                }
                else throw new IOException("ndarray type of Y not supported");
            }

        }
        return dataY;
    }
}
