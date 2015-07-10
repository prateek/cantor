package com.cloudera.cantor.pipeline

import java.util.UUID

/**
 * Created by prungta on 7/9/15.
 */
object Identifiable {
  def randomUID(prefix: String): String = {
    prefix + "_" + UUID.randomUUID().toString.takeRight(12)
  }
}
