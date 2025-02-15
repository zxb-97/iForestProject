# iForest in java
This is an implementation of the iForest algoritm in java. The project, contains a parser for .npy files and also for .npz files
that converts those formats to java arrays.

## Requirements
For evaluating the model with AUCROC and AUCPR scores, I've used the jstacs library. The jstacs-2.3.jar file 
can be found at this [link](https://www.jstacs.de/index.php/Downloads). Once downloaded the file must be placed in 
the build/libs directory (assuming you're using Intellij). To set it up, add the following to your `build.gradle`:

```gradle
repositories {
    mavenCentral()
    flatDir {
        dirs 'build/libs'  // Point to the correct directory
    }
}

dependencies {
    // Using local JAR file
    implementation files('build/libs/jstacs-2.3.jar')
}
```

