
// factor out common settings into a sequence
lazy val commonSettings = Seq(
  name := "dropout",
  organization := "dropout",
  version := "0.1.0",
  // set the Scala version used for the project
  scalaVersion := "2.10.4"
)


lazy val dropout = (project in file(".")).
  settings(commonSettings: _*).
  settings(libraryDependencies ++= Seq(
      // other dependencies here
      "org.apache.spark" % "spark-core_2.10" % "1.3.0",
      "org.apache.spark" % "spark-sql_2.10" % "1.3.0",
      "org.apache.spark" % "spark-mllib_2.10" % "1.3.0"
    )
  )